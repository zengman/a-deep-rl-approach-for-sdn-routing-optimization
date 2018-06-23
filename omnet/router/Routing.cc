//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
//

#include "Routing.h"
#include <ctime>
long getCurrentTime()
{
   struct timeval tv;
   gettimeofday(&tv,NULL);
   return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}
Define_Module(Routing);


Routing::Routing() {
}

Routing::~Routing() {
}

// -12, number of nodes
void Routing::initialize()
{
    numPorts = gateSize("in");
    id = par("id");
    numTx = par("numTx");
    numNodes = par("numNodes");
    folderName = par("folderName").stdstringValue();
    flow_num = par("flow_num");
    outPort ;

    int diff = numNodes - numTx;
//    long long t_start =getCurrentTime();
//
//
//    for (int i = 0; i < flow_num; i++){
//        int outPort_f[100];
//        getRoutingInfo(id-diff, i,outPort_f);
//    }
//
//    long long t_end = getCurrentTime();
    //cout<<"routing time txt = " << t_end - t_start<<endl;

    Statistic::instance()->setNumNodes(numNodes);
    Statistic::instance()->setNumFlow(flow_num);
    Statistic::instance()->setNumTx(numTx);
    Statistic::instance()->setFolder(folderName);
    Statistic::instance()->getRoutingById(id-diff, &outPort);
    //Statistic::instance()->getRoutingInfoAllFlows(&outPort, id-diff, i, outPort_f);


}

void Routing::handleMessage(cMessage *msg)
{
    DataPacket *data = check_and_cast<DataPacket *>(msg);
    double timeout=0.5;
    if (id == data->getDstNode()) {
        ev << this->getFullPath() << "  Message received" << endl;
        simtime_t delayPaquet= simTime() - data->getCreationTime();
        // 传递当前时间， 源节点，目的节点，包的大小，抵达的时间，流的编号
        int ttl = numNodes - data->getTtl() ;
        double jitter = 0.01;
        Statistic::instance()->setDelay(simTime(), data->getSrcNode(), data->getDstNode(), delayPaquet.dbl() + ttl * jitter,data->getFlow_id(), data->getBitLength());
        delete msg;
    }
    else if (data->getTtl() == 0 ) {
        ev << this->getFullPath() << "  TTL = 0. Msg deleted" << endl;
        //cout<<"time out flow_id="<<data->getFlow_id()<<",src="<<data->getSrcNode()<<",dst="<< data->getDstNode()<<endl;
        Statistic::instance()->setLost(simTime(), data->getSrcNode(), data->getDstNode(), data->getFlow_id(), data->getBitLength());
        delete msg;
    }
    else { // Tant in com out
        int flow_id = data->getFlow_id();
        int destPort = outPort[flow_id][data->getDstNode()];
        
        //ev<<"flow_id="<<flow_id<<"destPort="<<destPort<<",src="<<data->getSrcNode()<<",dst="<< data->getDstNode()<<endl;
        data->setTtl(data->getTtl()-1);
        send(msg, "out", destPort);
        ev << "Routing: " << this->getFullPath() << "  Source: " << data->getSrcNode() << " Dest: " << data->getDstNode()
           << " using port: "<< destPort << endl;
    }
    //if (msg->arrivedOn("localIn")) {

}

void Routing::getRoutingInfo(int id, int flow_id, int rData[]) {
//     for (int x =0; x < flow_num; x++){
         int x = flow_id;
         stringstream stream;
         stream<<x;
         string string_temp=stream.str();
         string txtName = "/Routing" + string_temp + ".txt";
         ifstream myfile (folderName + txtName);
         double val;

              if (myfile.is_open()) {
                  int i = 0;
                  while (id != i) {
                      for(int k = 0; k < numTx; k++) {
                          string aux;
                          getline(myfile, aux, ',');
                      }
                      //myfile >> val;
                      i++;
                  }

                  for(int k = 0; k < numTx; k++) {
                      string aux;
                      getline(myfile, aux, ',');
                      val = stod(aux);
                      rData[k] = val;
                      outPort[flow_id][k] = val;
                  }

                  myfile.close();
              }
//     }




}
