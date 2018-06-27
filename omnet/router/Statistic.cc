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
#include <ctime>
//#include <OutPortVector.h>
using namespace std;
void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
  std::string::size_type pos1, pos2;
  pos2 = s.find(c);
  pos1 = 0;
  while(std::string::npos != pos2)
  {
    v.push_back(s.substr(pos1, pos2-pos1));

    pos1 = pos2 + c.size();
    pos2 = s.find(c, pos1);
  }
  if(pos1 != s.length())
    v.push_back(s.substr(pos1));
}
long getTime()
{
   struct timeval tv;
   gettimeofday(&tv,NULL);
   return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}
Statistic *Statistic::inst = 0;

Statistic::Statistic() {

    INI = 50;
    END = 650;
    collect = true;
    flag_for_isReadRouting = false;
    flag_for_isReadTraffic = false;
    Traffic =  vector<vector<vector<double> > > (100, vector<vector<double> >(100, vector<double>()));
    Routing =  vector<vector<double>  > (100, vector<double>(100));
//    BandWidth = vector<vector<double>  > (100, vector<double>(100, 0));
//    Delay =  vector<vector<vector<double> > > (100, vector<vector<double> >(100, vector<double>()));
    Delay = vector<vector<vector<vector<double > > > >(100, vector<vector<vector<double>>>(100,vector<vector<double>>(100,vector<double>())));
    BandFlowPathPacketsSize = vector<vector<vector<long int>>>(100, vector<vector<long int>>(100,vector<long int>(100,0)));
    // vector<vector<vector<double> > > (100, vector<vector<double> >(100, vector<double>()));

//    DropsV =  vector<vector<double>  > (100, vector<double>(100, 0));
    Jitter = vector<vector<vector<vector<double > > > >(100, vector<vector<vector<double>>>(100,vector<vector<double>>(100,vector<double>())));
    DropsV = vector<vector<vector<double> > > (100, vector<vector<double> >(100, vector<double>(100,0)));
    drops = 0;
    flow_id = 0;
    Numpackets =  vector<long int>(100,0);
    BandPakcetsSize = vector<long int>(100,-1); // 记录每条流发送的所有size的大小,路径上实际发送的size内容
    SendPackets =  vector<long int>(100,0);

    outportlist = vector<vector<vector<int>>> (100, vector<vector<int>> (100, vector<int>(100,-1)));
    trafficlist = vector<vector<double>>(100, vector<double>(100,0));

}

Statistic::~Statistic() {

}



Statistic *Statistic::instance() {
    if (!inst)
      inst = new Statistic();
    return inst;
}


void Statistic::getRoutingById(int id, vector<vector<int> > *outPort){
    getRoutingInfo(&outportlist);
    (*outPort).swap(outportlist[id] );
//    if(id==18){
//            cout<<"static outport "<<(*outPort)[13][21]<<endl;
//        }
//    if(id==3)cout<<"checkt node 3 start "<<(*outPort)[44][25]<<endl;
}
bool Statistic::appShouldSend(int flow_id, int src, int dest){
    // 判断该包是否在所有流的src dst，如果是的话，发送，不是的话，删除该包，且不予发送
    // FLow_path 第一个元素 是长度 不是path内容!
    if(src == Flow_info[flow_id][0]&& dest == Flow_info[flow_id][1]){
       // ev<<"shoule send "<< Flow_info[flow_id][0]<<" dest"<<Flow_info[flow_id][1]<<endl;
        return true;
    }
    return false;

}

void Statistic::getTrafficInfo(int flow_id, double *bandwidth, int *tsrc, int *tdest, double *bandwidth_df) {

    if(flag_for_isReadTraffic == true){
        *bandwidth = trafficlist[flow_id][0];
        *tsrc = trafficlist[flow_id][1];
        *tdest = trafficlist[flow_id][2];
        *bandwidth_df = trafficlist[flow_id][3];
        return;
    }else{
        flag_for_isReadTraffic = true;
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
                        double bw_df = stod(aux);
                        getline(myfile, aux,',');//DF
                        double bw = stod(aux);
                        //cout<<"DF="<<bw<<",flow_num="<<i<<endl;

                        trafficlist[i][0] = bw;
                        trafficlist[i][1] = src;
                        trafficlist[i][2] = dst;
                        trafficlist[i][3] = bw_df;
                    }

                    *bandwidth = trafficlist[flow_id][0];
                     *tsrc = trafficlist[flow_id][1];
                      *tdest = trafficlist[flow_id][2];
                      *bandwidth_df = trafficlist[flow_id][3];
                    //cout<<"flow_id="<<flow_id<<"statit bw="<<*bandwidth<<",src"<<*src<<",dest="<<*dest<<endl;
                }
                myfile.close();

    }

}

void Statistic::getRoutingInfo(vector<vector<vector<int>>> *outportlist) {
    // 读取所有Routing.txt文件，得到flow个数和，节点个数
       // 只读取一次

       if(flag_for_isReadRouting == true){

           return;
       }

       flag_for_isReadRouting = true;
       cout<<"read routing.txt"<<endl;
       ifstream readpath (folderName + "/flowpath.txt");

       Flow_info = vector<vector<int>  > (100, vector<int>(100));
       FLow_path = vector<vector<int>> (100, vector<int>(20,-1));

         if(readpath.is_open()){
             string str;
             for(int i=0; i< flow_num;i++){
                 getline(readpath,str);
                 vector<string> v ;
                 SplitString(str, v,",");// 节点， 端口, 节点， 端口， 。。。 节点
                 Flow_info[i][0]=  stoi(v[0]); // 源 第一个元素
                 Flow_info[i][1] = stoi(v[v.size()-1]);// 目的 最后一个元素
//                 cout<<"path size="<<v.size()<<endl;

                 int cnt_for_path = 1;
                 int j = 0;
                 while(j + 2 < v.size()){
                     int src = stoi(v[j]);
                     int port = stoi(v[j+1]);
                     int dst = stoi(v[j+2]);
                     //if(i==0)cout<<"flow_num"<<i<<" s:"<<src<< " p:"<<port<< " dst:"<< dst<<endl;
                     FLow_path[i][cnt_for_path++] = src;
                     //FLow_path[i][cnt_for_path++] = dst;
                     (*outportlist)[src][i][Flow_info[i][1]] = port;// dest = 为最后的节点
                     //if(src==6 && dst== 9)cout<<"6-9 port="<<port<<endl;
                     j += 2;
                 }
                 FLow_path[i][cnt_for_path] = Flow_info[i][1];
                 FLow_path[i][0] = cnt_for_path;

             }
             //cout<<"checkt node 3 last"<< (*outportlist)[3][44][25]<<endl;
         }
         readpath.close();

}


void Statistic::setMaxSim(double ms) {
    END = ms;
    SIMTIME = (END.dbl()-INI.dbl())*1000;
}

void Statistic::setRouting(int src, int dst, double r) {
    (Routing)[src][dst] = r;

}

void Statistic::infoTS(simtime_t time, int numpackets, int flow_id, int size) {
    if (time > END and collect) {
        collect = false;
        cout<<"info"<<endl;
        printStats(time);

    }
    if (time < INI and not collect){
        collect = true;


    }



//        cout<<"all size="<<Numpackets[flow_id]<<endl;

}

 void Statistic::setDelay(simtime_t time, int src, int dst, double d, int flow_id, int size) {
     //传递当前时间， 源节点，目的节点，，抵达的时间s，流的编号
     if (time > INI and collect){
         (Delay)[flow_id][src][dst].push_back(d);
         //Numpackets[flow_id] ++;
         SendPackets[flow_id] ++;
         //BandPakcetsSize[flow_id]+=size;
         BandFlowPathPacketsSize[flow_id][src][dst] += size;
     }
 }

void Statistic::setTraffic(simtime_t time, int src, int dst, double t) {
    if (time > INI and collect){
        (Traffic)[src][dst].push_back(t);
    }
}

void Statistic::setLost(simtime_t time, int n, int p, int flow_id, int size) {
    if (time > INI and collect) {
//        drops++;
        (DropsV)[flow_id][n][p]++;
        //BandPakcetsSize[flow_id]-=size;
        Numpackets[flow_id] ++; //记录丢包的个数
        //cout<<"lost paket number="<<Numpackets[flow_id]<<endl;
    }
}

void Statistic::setLost(simtime_t time) {
    if (time > INI and collect){
        drops++;

    }

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

void Statistic::setFlowId(int n){
    flow_id = n;
}


void Statistic::printStats(simtime_t timet) {

   long long t_start =getTime();


    vector<double> features;


     // Delay
    for (int i = 0; i < flow_num; i++) {
//       for (int j = 0; j < numTx; j++) {
           int src = Flow_info[i][0];
           int dst = Flow_info[i][1];

           long double d = 0;
           //long int maxPacket = BandFlowPathPacketsSize[i][FLow_path[i][1]][FLow_path[i][2]];
           //BandPakcetsSize[i] = maxPacket;
//           unsigned int numPackets = 0;
//           for(int j=1; j<FLow_path[i][0]; j++){
//               int tsrc = FLow_path[i][j];
//               int tdst = FLow_path[i][j+1];
//               if(maxPacket>=BandFlowPathPacketsSize[i][tsrc][tdst]){
//                   BandPakcetsSize[i] = BandFlowPathPacketsSize[i][tsrc][tdst];
//                   maxPacket=BandPakcetsSize[i] ;
//                   //cout<<"flow_Id="<<i<<", maxPacket="<<maxPacket<<endl;
//                   //选取一条流的实际是哪个传输的最小的size
//               }
//              // cout<<"start size="<<FLow_path[i][0]<<",i="<<i<<",j="<<j<<endl;
//               long int pernumPackets = (Delay)[i][tsrc][tdst].size(); // 每一条流的一条路径上每一跳发送的包数总和
//               //cout<<"perpakerct="<<pernumPackets<<", src="<<tsrc<<",dst="<<tdst<<endl;
//               numPackets += pernumPackets;
//               for (unsigned int k = 0; k < pernumPackets; k++){
//                   d += (Delay)[i][tsrc][tdst][k]; // 获取延迟包delay的时间
//                }
//           }
           unsigned int numPackets=(Delay)[i][src][dst].size(); // 源到目的传送了多少个包
           for (unsigned int k = 0; k < numPackets; k++){
               d += (Delay)[i][src][dst][k]; // 获取延迟包delay的时间
               BandPakcetsSize[i] = BandFlowPathPacketsSize[i][src][dst];
           }
           //cout<<"delay numbpakets = " << numPackets<<", src="<<src<<",dst="<<dst<<endl;

           //cout<<"delay d="<< d<<endl;
           if (numPackets == 0)
               if (src == dst)
                   features.push_back(-1);
               else{
                   features.push_back(0);
                }
              
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
        d = d * 1000  ; // delay 的延迟时间单位是秒， 写入时，转换为毫秒
        myfile_delay  << d << ",";
    }
    myfile_delay << endl;
    myfile_delay.close();


  // Drops
    vector<double> features2;
    for (int i = 0; i < flow_num; i++) {

           long double lostp = Numpackets[i];  //kb
           
           long double sendp = SendPackets[i] + lostp; // 总的发送包数目
           if(sendp == 0){
               features2.push_back(0);
               continue;
           }
           long double rate =lostp/sendp;
           
//           cout<<"sendp = "<<sendp<<", lostp="<<lostp<<endl;
           features2.push_back(rate);
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
        int src = Flow_info[i][0];
        int dst = Flow_info[i][1];
        long double d = 0;

//        unsigned int numPackets = 0;
//        for(int j=1; j<FLow_path[i][0]; j++){
//            int tsrc = FLow_path[i][j];
//            int tdst = FLow_path[i][j+1];
//            long int pernumPackets = (Delay)[i][tsrc][tdst].size(); // 每一条流的一条路径上每一跳发送的包数总和
//            numPackets += pernumPackets;
//            for (unsigned int k = 0; k < pernumPackets-1; k++){
//                d += abs((Delay)[i][tsrc][tdst][k+1] - (Delay)[i][tsrc][tdst][k]); // 获取延迟包delay的时间
//            }
//
//        }
        unsigned int numPackets = (Delay)[i][src][dst].size();
        if (numPackets == 0){
            if (src == dst)
                features3.push_back(-1);
            else {
                features3.push_back(0);
            }
            continue;
        }
        for(int k=0; k<numPackets-1; k++){
            d += abs( (Delay)[i][src][dst][k+1] - (Delay)[i][src][dst][k]);
        }
        if (numPackets == 1)
            features3.push_back(0);
        else
        {
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
        //相当于jitter 基础是10000
        d = d * 1000 * 10 ; //delay 的延迟时间单位是秒， 写入时，转换为毫秒
        myfile3  << d << ",";
    }
    myfile3 << endl;
    myfile3.close();

    // bandwidth
         vector<double> features_bw;
         for (int i = 0; i < flow_num; i++) {
      //       for (int j = 0; j < numTx; j++) {
                long double datalength  =0;
   //             = BandPakcetsSize[i]/1024; // bandPacetsize 单位为bit，bandwidth单位为kbps
                //cout<<"bandwidth packets="<<BandPakcetsSize[i]<<",timet="<<timet<<endl;
//                cout<<(1-features2[i])<<endl;
                datalength = (1-features2[i]) * trafficlist[i][3]; // 实际跑的带宽等于丢包率 * 发送带宽
                if (datalength == 0)
                    features_bw.push_back(0);
                else
                    features_bw.push_back(datalength); // 带宽= kbps, timet 为ms
      //       }
         }
         ofstream myfile_bw;
         string filename_bw;
         filename_bw = folderName + "/Bandwidth.txt";
         myfile_bw.open (filename_bw, ios::out | ios::trunc );
         for (unsigned int i = 0; i < features_bw.size(); i++ ) {
             double d = features_bw[i];
             myfile_bw  << d << ",";
         }
         myfile_bw << endl;
         myfile_bw.close();


   // Reset
    drops = 0;
    Traffic.clear();
    Delay.clear();
    DropsV.clear();
    Jitter.clear();
    SendPackets.clear();
    Numpackets.clear();
    features_bw.clear();
    features3.clear();
    features2.clear();
    BandFlowPathPacketsSize.clear();
    BandPakcetsSize.clear();





    long long t_end = getTime();
    cout<<"statistic time txt = " << t_end - t_start<<endl;

}
