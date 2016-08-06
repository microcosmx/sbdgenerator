import akka.actor._
import akka.routing._
import org.apache.spark._

object Actors {
    lazy val remoteAddr = RemoteAddressExtension(ActorSystem("sbd")).address
    def remotePath(actor:ActorRef) = actor.path.toStringWithAddress(remoteAddr)
}

class RemoteAddressExtensionImpl(system: ExtendedActorSystem) extends Extension {
    def address = system.provider.getDefaultAddress
}

object RemoteAddressExtension extends ExtensionKey[RemoteAddressExtensionImpl]
