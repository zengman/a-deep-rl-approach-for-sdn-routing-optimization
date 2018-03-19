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

        int size;
//        genT = 0;
        switch (genT) {
            case 0: // Poisson
                size = exponential(10000);
                if (size > 73728) size = 73728;
//                if(size < 4096) size = 4096;

//                if (size < 4096) size = 4096;
                break;
            case 1: // Deterministic
                size = 5120;
                break;
            case 2: // Uniform
                size = uniform(4096,73728);
                break;
            case 3: // Binomial
                if (dblrand() < 0.5) size = 300;
                else size = 1700;
                break;
            default:
                break;
        }
//        size = size * 64 * 16 * 2 + 2048 + 1024;
//        if (size > 73728) size = 73728;
        TimerNextPacket *time =  check_and_cast<TimerNextPacket *>(msg);
        int flow_id = time->getFlow_id();
        double bandwidth = time->getBandwidth();
        data->setBitLength(size);
        data->setTtl(numRx);
        data->setDstNode(dest);
        data->setSrcNode(id);
        data->setLastTS(simTime().dbl());
        data->setFlow_id(flow_id);
        data->setBandwidth(bandwidth);

//        ev << "now time" << simTime() << "----packets" << numPackets
//                            << "send packet: "<< id << "----"<< dest << endl;
        send(data, "out");

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
//                etime = exponential(1.0/bandwidth);
//                etime = (1/bandwidth);
                etime = size/(bandwidth*1024*1024);

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
        lambda = lambdaFactor*flowRatio;

        //lambda = lambdaMax/numRx;
        if(lambda == 0) {
            lambda = 1;
        }
        interArrival = new TimerNextPacket("timer");
        interArrival->setLambda(1.0/lambda);
        interArrival->setFlow_id(flow_id);
        interArrival->setBandwidth(flowRatio);
        if (dest != id)
            scheduleAt(simTime() + 1.0/lambda, interArrival);
            ev << "Ratio: " << flowRatio << "   lambda: " << lambda << endl;

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
