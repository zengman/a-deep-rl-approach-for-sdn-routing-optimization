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

#include <Statistic.h>
#include <limits>
#include <cmath>

Statistic *Statistic::inst = 0;



Statistic::Statistic() {

    INI = 50;
    END = 650;
    collect = true;

    Traffic =  vector<vector<vector<double> > > (100, vector<vector<double> >(100, vector<double>()));
    Routing =  vector<vector<double>  > (100, vector<double>(100));
//    BandWidth = vector<vector<double>  > (100, vector<double>(100, 0));
    Delay =  vector<vector<vector<double> > > (100, vector<vector<double> >(100, vector<double>()));
    // vector<vector<vector<double> > > (100, vector<vector<double> >(100, vector<double>()));
    DropsV =  vector<vector<double>  > (100, vector<double>(100, 0));
    Jitter = vector<vector<vector<double> > > (100, vector<vector<double> >(100, vector<double>()));
    drops = 0;



}

Statistic::~Statistic() {

}



Statistic *Statistic::instance() {
    if (!inst)
      inst = new Statistic();
    return inst;
}

void Statistic::setMaxSim(double ms) {
    END = ms;
    SIMTIME = (END.dbl()-INI.dbl())*1000;
}

void Statistic::setRouting(int src, int dst, double r) {
    (Routing)[src][dst] = r;
}

void Statistic::infoTS(simtime_t time) {
    if (time > END and collect) {
        collect = false;
        printStats();
    }
    if (time < INI and not collect)
        collect = true;
}

//void Statistic::setDelay(simtime_t time, int flow_id, double d) {
//    if (time > INI and collect)
//        (Delay)[flow_id].push_back(d);
//}

 void Statistic::setDelay(simtime_t time, int src, int dst, double d) {
     if (time > INI and collect)
         (Delay)[src][dst].push_back(d);
 }

// jitter ����delay�����ˣ���˲������
// void Statistic::setJitter(simtime_t time, int src, int dst, double d) {
//     if (time > INI and collect)
//         (Jitter)[src][dst].push_back(k);
// // }
// void Statistic::setJitter(simtime_t time, int flow_id, double d) {
//     if (time > INI and collect)
//         (Jitter)[flow_id].push_back(d);
// }


void Statistic::setTraffic(simtime_t time, int src, int dst, double t) {
    if (time > INI and collect)
        (Traffic)[src][dst].push_back(t);
}

void Statistic::setLost(simtime_t time, int n, int p) {
    if (time > INI and collect) {
        drops++;
        (DropsV)[n][p]++;
    }
}

void Statistic::setLost(simtime_t time) {
    if (time > INI and collect)
        drops++;
}


/*void Statistic::setRouting(int n, int r, double p) {
}*/

void Statistic::setLambda(double l) {
    lambdaMax = l;
}

void Statistic::setGeneration(int genType) {
    genT = genType;
}

void Statistic::setFolder(string folder) {
    folderName = folder;
}



void Statistic::setNumTx(int n) {
    numTx = n;
}

void Statistic::setNumFlow(int n) {
    flow_num = n;
}

void Statistic::setNumNodes(int n) {
    numNodes = n;
}

void Statistic::setRoutingParaam(double r) {
    routingP = r;
}


void Statistic::printStats() {
    string genString;
    switch (genT) {
        case 0: // Poisson
            genString = "M"; //"Poisson";
            break;
        case 1: // Deterministic
            genString = "D"; //"Deterministic";
            break;
        case 2: // Uniform
            genString = "U"; //"Uniform";
            break;
        case 3: // Binomial
            genString = "B"; //"Binomial";
            break;
        default:
            break;
    }

    vector<double> features;
//    features.push_back(routingP);

    //int firstTx = numNodes - numTx;
    // Traffic
    /*for (int i = 0; i < numTx; i++) {
       for (int j = 0; j < numTx; j++) {
           long double d = 0;
           unsigned int numPackets = (Traffic)[i][j].size();
           for (unsigned int k = 0; k < numPackets; k++)
               d += (Traffic)[i][j][k];
           features.push_back(d/SIMTIME);
       }
    }
    // Routiung
    for (int i = 0; i < numTx; i++) {
       for (int j = 0; j < numTx; j++) {
           features.push_back((Routing)[i][j]);
       }
    }*/


// /*     // Delay origin
//     int steps = (SIMTIME/1000)+50;
//     for (int i = 0; i < numTx; i++) {
//        for (int j = 0; j < numTx; j++) {
//            long double d = 0;
//            unsigned int numPackets = (Delay)[i][j].size();
//            for (unsigned int k = 0; k < numPackets; k++)
//                d += (Delay)[i][j][k];
//            if (numPackets == 0)
//                if (i == j)
//                    features.push_back(-1);
//                else
//                    features.push_back(std::numeric_limits<double>::infinity());
//            else
//                features.push_back(d/numPackets);
//        }
//     } */
    // Drops
//     for (int i = 0; i < numTx; i++) {
//        for (int j = 0; j < numTx; j++) {
//            features.push_back((DropsV)[i][j]/steps);
//        }
//     }
    // features.push_back(drops/steps);


    ifstream myfile (folderName + "/Traffic.txt");
    Flow_info = vector<vector<int>  > (100, vector<int>(100));
    vector<double> df;
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
            Flow_info[i].push_back(src);
            Flow_info[i].push_back(dst);
            df.push_back(bw);
            cout<<"bw="<<bw<<endl;
        }
    }
    myfile.close();


    ofstream myfile_flow;
    string filenameflow;
    filenameflow = folderName + "/Flow.txt";
    myfile_flow.open (filenameflow, ios::out | ios::trunc);
    for (unsigned int i=0; i< flow_num; i++){
        myfile_flow << i << ":";
        myfile_flow << Flow_info[i][0] << ","<< Flow_info[i][1]<<','<<df[i]<<',';

    }
    myfile_flow << endl;
    myfile_flow.close();

     // Delay
    int steps = (SIMTIME/1000)+50;
    for (int i = 0; i < flow_num; i++) {
//       for (int j = 0; j < numTx; j++) {
           int src = Flow_info[i][0];
           int dst = Flow_info[i][1];
           cout<< "src = "<<src<<";dst="<<dst<<endl;
           long double d = 0;
           unsigned int numPackets = (Delay)[src].size();
           for (unsigned int k = 0; k < numPackets; k++)
               d += (Delay)[src][dst][k];
           if (numPackets == 0)
               if (src == dst)
                   features.push_back(-1);
               else
                   features.push_back(std::numeric_limits<double>::infinity());
           else
               features.push_back(d/numPackets);
//       }
    }

    // Print file
    ofstream myfile_delay;
    string filename;
    // Instant
    filename = folderName + "/Delay.txt";
    myfile_delay.open (filename, ios::out | ios::trunc );
    for (unsigned int i = 0; i < features.size(); i++ ) {
        double d = features[i];
        myfile_delay  << d << ",";
    }
    myfile_delay << endl;
    myfile_delay.close();

//    // Bandwidth
//    vector<double> features_bw;
//    for (int i = 0; i < numTx; i++) {
//    //    for (int j = 0; j < numTx; j++) {
//           features_bw.push_back((Bandwidth)[i]/steps);
//    //    }
//    }

    // features_bw.push_back(drops/steps);
//    ofstream myfile_bw;
//    string filename_bw;
//    filename_bw = folderName + "/Bandwidth.txt";
//    myfile_bw.open (filename_bw, ios::out | ios::trunc );
//    for (unsigned int i = 0; i < features_bw.size(); i++ ) {
//        double d = features_bw[i];
//        myfile_bw  << d << ",";
//    }
//    myfile_bw << endl;
//    myfile_bw.close();

  // Drops
    vector<double> features2;
    for (int i = 0; i < flow_num; i++) {
    //    for (int j = 0; j < numTx; j++) {
           int src = Flow_info[i][0];
           int dst = Flow_info[i][1];
           features2.push_back((DropsV)[src][dst]/steps);
    //    }
    }
    // features2.push_back(drops/steps);
    ofstream myfile2;
    string filename2;
    filename2 = folderName + "/PacketLossRate.txt";
    myfile2.open (filename2, ios::out | ios::trunc );
    for (unsigned int i = 0; i < features2.size(); i++ ) {
        double d = features2[i];
        myfile2  << d << ",";
    }
    myfile2 << endl;
    myfile2.close();


 // Jitter
    vector<double> features3;
    Jitter = Delay;
    for (int i = 0; i < flow_num; i++) {
    //    for (int j = 0; j < numTx; j++) {
        int src = Flow_info[i][0];
        int dst = Flow_info[i][1];
        long double d = 0;
        unsigned int numPackets = (Jitter)[src][dst].size();
        if (numPackets == 0)
            if (src == dst)
                features3.push_back(-1);
            else
                features3.push_back(std::numeric_limits<double>::infinity());
        else if (numPackets == 1)
            features3.push_back(0);
        else
        {
            for (unsigned int k = 0; k < numPackets-1; k++)
                d += abs((Jitter)[src][dst][k+1] - (Jitter)[src][dst][k]);
            features3.push_back(d/(numPackets-1));
        }
    //    }
    }

    ofstream myfile3;
    string filename3;
    filename3 = folderName + "/Jitter.txt";
    myfile3.open (filename3, ios::out | ios::trunc );
    for (unsigned int i = 0; i < features3.size(); i++ ) {
        double d = features3[i];
        myfile3  << d << ",";
    }
    myfile3 << endl;
    myfile3.close();

   // Reset
    drops = 0;
    Traffic.clear();
    Delay.clear();
    DropsV.clear();
    // Jitter.clear();

}
