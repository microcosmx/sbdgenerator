import org.apache.hadoop.fs._
import resource._
import scala.concurrent._
import scala.util.control._
import org.apache.spark.rdd._
import scala.concurrent._
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._

object HadoopConversions {
    implicit def remoteIteratorAsStream[T](i:RemoteIterator[T]):Stream[T] =
        if (i.hasNext) { i.next #:: remoteIteratorAsStream(i) } else Stream()
    implicit def stringAsPath(path: String) = new Path(path)
    implicit def pathAsString(path: Path) = path.toString

    implicit def pathOps(x: String) = new {
        def /(y:String) = x + "/" + y
        def toAbsolutePath = (new java.io.File(x)).getAbsolutePath
        def toCanonicalPath = (new java.io.File(x)).getCanonicalPath
    }

    implicit def fsOps(fs: FileSystem) = new {
        def mkTempPath(path: Path, extension: String = "temp") : Path =
            new Path(path.getParent, s".${path.getName}-${System.nanoTime}.$extension")

        def textFile[T](path: String)(f: scala.io.Source => T): T =
            blocking { managed(scala.io.Source.fromInputStream(fs.open(path))).acquireAndGet(f) }

        def writeFileAtomic[T](dstPath: String)(f: String => T): T = {
            val tmpPath = mkTempPath(dstPath)

            val result = try blocking { f(tmpPath) }
            catch {
                // scala sometimes uses exceptions for normal control flow
                case t: ControlThrowable => throw t
                case t: Throwable =>
                    try { blocking { fs.delete(tmpPath, true) } }
                    catch { case _ : Throwable => }
                    throw t
            }

            blocking {
                if (fs.exists(dstPath)) {
                    val delPath = mkTempPath(dstPath, "delete")
                    // minimize time when file system is in inconsistent state
                    // don't leave file artifacts incomplete even during delete
                    fs.rename(dstPath, delPath)
                    fs.rename(tmpPath, dstPath)
                    fs.delete(delPath, true)
                }
                else
                    fs.rename(tmpPath, dstPath)
            }

            result
        }

        def writeFile[T](dstPath: String)(f: FSDataOutputStream => T): T =
            writeFileAtomic(dstPath) { path =>
                managed(fs.create(path, true)).acquireAndGet(f)
            }

        def saveAsTextFile(dstPath: String, content: String) =
            writeFile(dstPath)(_.write(content.getBytes))

        def saveAsTextFile(dstPath: String, lines: Iterable[String]) =
            writeFile(dstPath) { stream =>
                lines foreach { line =>
                    stream.write(line.getBytes)
                    stream.write("\n".getBytes)
                }
            }
        
        def writeFilesRDD(rddFinal: Map[String,RDD[String]], files: Map[String,String], defaultRDD: RDD[String]) = {
            val futureResult = Future.sequence {
                files.map {
                    case (key, path) => future { 
                      blocking { 
                        if(rddFinal.contains(key)){
                          writeFileRDD(rddFinal(key), path)
                        }else{
                          writeFileRDD(defaultRDD, path)
                        }
                      }
                    }
                }
            }
            Await.result(futureResult, Duration.Inf)
        }
        
        def writeFileRDD(rdd: RDD[String], targetPath: String) = {
            //println("------rdd writing-----")
            val timestamp = System.nanoTime
            val src = s"${targetPath}-src-${timestamp}"
            val dst = s"${targetPath}-dst-${timestamp}"
            val srcPath = new Path(src)
            val dstPath = new Path(dst)
            val dstFinalPath = new Path(targetPath)
            try blocking { 
              rdd.saveAsTextFile(src) 
              FileUtil.copyMerge(fs, srcPath, fs, dstPath, true, fs.getConf ,null)
            }
            catch {
                case t: ControlThrowable => throw t
                case t: Throwable =>
                    try { blocking { 
                      fs.delete(srcPath, true) 
                      fs.delete(dstPath, true) 
                    } }
                    catch { case _ : Throwable => }
                    throw t
            }
            
            blocking {
              if (fs.exists(dstFinalPath)) {
                  val delPath = mkTempPath(dstFinalPath, "delete")
                  fs.rename(dstFinalPath, delPath)
                  fs.rename(dstPath, dstFinalPath)
                  fs.delete(delPath, true)
              }
              else
                  fs.rename(dstPath, dstFinalPath)
            }
            //println("------rdd end-----")
        }
    }
}
