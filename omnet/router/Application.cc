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

#include "Application.h"

// Application 有一个schedule， 设置好包的内容，在下一个时间间隔发给自己处理
Define_Module(Application);


Application::Application() {

     interArrival = NULL;

}

Application::~Application() {
     cancelAndDelete(interArrival);
}


void Application::initialize()
{
    //id = extractId(this->getFullPath(), 12);
    id = par("id");
    genT = par("generation");
    lambdaFactor = par("lambda");
    dest = par("dest");
    MAXSIM = par("simulationDuration");
    numRx = par("numNodes");
    numPackets = 0;

    Statistic::instance()->setGeneration(genT);
    Statistic::instance()->setMaxSim(MAXSIM);
    Statistic::instance()->setLambda(lambdaFactor);


}


void Application::handleMessage(cMessage *msg)
{
    if (msg->isSelfMessage()) {

        DataPacket *data = new DataPacket("dataPacket");
        TimerNextPacket *time =  check_and_cast<TimerNextPacket *>(msg);
        int flow_id = time->getFlow_id();
        double bandwidth = time->getBandwidth();
        double bandwidth_df = time->getBandwidth_df();

        int size;
        size = size = exponential(10000);
        if(size>7000){
            size=7000;
        }
        data->setBitLength(size);
        data->setTtl(numRx); // 最大跳数
        data->setDstNode(dest);
        data->setSrcNode(id);
        data->setLastTS(simTime().dbl());// 最后一次的发送时间？？
        data->setFlow_id(flow_id);
        data->setBandwidth(bandwidth);

        send(data, "out");
        //cout<<"app send paket flow_id="<<flow_id<<", sr="<<id<<",dest="<<dest<<endl;
        numPackets++;
        Statistic::instance()->setTraffic(simTime(), id, dest, size);
        Statistic::instance()->infoTS(simTime(), numPackets, flow_id,size);

        if (simTime() < MAXSIM) {

            simtime_t etime;
            if(bandwidth == 0.01 || bandwidth == 0) {
                etime = exponential(1.0/lambda);
//                ev << "normal packet: "<< id << "----"<< dest << endl;
            }
            else{
                double bd = uniform(bandwidth_df, bandwidth);
                double jitter = 0.5;
                etime = size/(bandwidth_df*1024); // 用df去跑

            }
            scheduleAt(simTime() + etime, msg);

//            ev << "now time" << simTime() << "----packets" << numPackets
//                    << "speed packet: "<< id << "----"<< dest <<":--"<<etime << endl;
        }
        else {
            EV << "END simulation" << endl;
        }
    }

    else {

        ControlPacket *data = check_and_cast<ControlPacket *>(msg);
        double flowRatio = data->getData();
        int flow_id = data->getFlow_id();
        double bandwidth_df = data->getBandwidth_df();

        bool isSend= Statistic::instance()->appShouldSend(flow_id,id,dest);
        //if(id==6 && dest==9) cout<<"6-9 app find="<<isSend<<endl;

        lambda = lambdaFactor*flowRatio;

        //lambda = lambdaMax/numRx;
        if(lambda == 0) {
            lambda = 1;
        }
        interArrival = new TimerNextPacket("timer");
        interArrival->setLambda(1.0/lambda);
        interArrival->setFlow_id(flow_id);
        interArrival->setBandwidth(flowRatio);
        interArrival->setBandwidth_df(bandwidth_df);
//        interArrival->
        // bandwidth 是traffic 读取的是df字段的内容，类似于现在流矩阵需要传送的数量
        if (dest != id && isSend == true){
            scheduleAt(simTime() + 1.0/lambda, interArrival);
            ev << "Ratio: " << flowRatio << "   lambda: " << lambda << "  id="<<id<<"   dest="<<dest<<endl;
        }


        delete data;


    }


}


/*
int Application::extractId(string name, int pos) {
    int idt;
    char c2 = name[pos+1]; char c1 = name[pos];
    if (c2 >= '0' and c2 <= '9')
        idt = (c1-'0')*10 + (c2 - '0');
    else
        idt = (c1-'0');
    return idt;
}*/
