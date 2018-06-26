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

#include "TrafficController.h"


Define_Module(TrafficController);

void TrafficController::initialize()
{
    id = par("id");
    nodeRatio = par("nodeRatio");
    numNodes = par("numNodes");
    folderName = par("folderName").stdstringValue();
    flow_num = par("flow_num");
    Flow_info = vector<vector<int>  > (100, vector<int>(100, 0));


    int MODE = 3;

    if (MODE == 1) {
        // UNIFORM TRAFFIC PER FLOW
        for (int i = 0; i < numNodes; i++) {
            double aux;
            if (i == id) aux = 0;
            else aux = uniform(0.1,1);

            ControlPacket *data = new ControlPacket("trafficInfo");
            data->setData(aux/numNodes);
            send(data, "out", i);

        }
    }
    else if (MODE == 2) {
        // UNIFOR TRAFFIC PER NODE
        double flowRatio[numNodes];
        double sumVal = 0;
        for (int i = 0; i < numNodes; i++) {
            double aux;

            if (i == id) aux = 0;
            else aux = uniform(0.1,1);

            flowRatio[i] = aux;
            sumVal += aux;
        }

        //ev << "NODE Ratio: " << nodeRatio << endl;
        for (int i = 0; i < numNodes; i++) {
            ControlPacket *data = new ControlPacket("trafficInfo");
            data->setData(nodeRatio*flowRatio[i]/sumVal);
            send(data, "out", i);
        }
    }
    else {
        // READED FROM FILE
//        getTrafficFromFile(Flow_info, Bandwidth);

        for (int j = 0; j < flow_num; j++){
            double bandwidth = 0;
            double bandwidth_df=0;
            int src=0,dest=0;
            Statistic::instance()->setNumNodes(numNodes);
            Statistic::instance()->setNumFlow(flow_num);
            Statistic::instance()->setFolder(folderName);
            Statistic::instance()->getTrafficInfo(j,&bandwidth, &src, &dest, &bandwidth_df);
            if(src == id   && bandwidth_df != 0){
                ControlPacket *data = new ControlPacket("trafficInfo");
                //cout<<"traffic flow_Id="<<j<<",src"<<src<<",dest="<<dest<<endl;
                data->setData(bandwidth);
                data->setBandwidth_df(bandwidth_df);
                data->setFlow_id(j);
                send(data, "out", dest);
            }

        }


    }

}

void TrafficController::handleMessage(cMessage *msg)
{
    // TODO - Generated method body
}


void TrafficController::getTrafficInfo(int id, int flow_id, double rData[]) {
     for(int i = 0; i < numNodes; i++){
         rData[i]=0.01;
     }
     rData[id]=-1;
     ifstream myfile (folderName + "/Traffic.txt");
         if(myfile.is_open()){
             for(int i = 0; i < flow_num; i++){
                 string aux;
                 getline(myfile, aux, ',');//flow_id
                 getline(myfile, aux, ',');//src
                 int src = stoi(aux);
                 getline(myfile, aux, ',');//dst
                 int dst = stoi(aux);
                 getline(myfile, aux,',');//df
                 double bw = stod(aux);
                 getline(myfile, aux,',');//Df
                 if (i == flow_id){
                     if (src == id){
                         rData[dst] = bw;
                     }
                     break;

                 }

             }
         }
         myfile.close();


}
