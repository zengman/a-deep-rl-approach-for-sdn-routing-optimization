package networksimulator;

network NetworkAll
{
    parameters:
        int numNodes = 35;
        int numTx = 35;

    types:
        channel Channel extends ned.DatarateChannel
        {
          // delay = 100μs;
           // double length @unit(m);
            //datarate = 25kbps;
            //ber = uniform(0.0001, 0.005);
            //delay = uniform(0.01ms, 0.1ms);
            //per = uniform(0.0001,0.005);
            //= this.length./200 000km* 1s;
            double jitter;
            jitter = uniform(0.01, 0.05);
            delay = 0.01s;
        }
        
        channel Channel_c extends ned.DatarateChannel
        {
            delay = 1ms;
        }

        submodules:
            sdnnode[numNodes]: Server {
                id = index;
                numTx = numTx;
                numNodes = numNodes;
                //gates:
                //    port[3];
            }

    connections:
        sdnnode[0].port++ <--> Channel_c{datarate=80kbps;} <--> sdnnode[24].port++;
        sdnnode[0].port++ <--> Channel_c{datarate=88kbps;} <--> sdnnode[30].port++;
        sdnnode[0].port++ <--> Channel_c{datarate=71kbps;} <--> sdnnode[32].port++;
        sdnnode[1].port++ <--> Channel_c{datarate=70kbps;} <--> sdnnode[3].port++;
        sdnnode[1].port++ <--> Channel_c{datarate=79kbps;} <--> sdnnode[6].port++;
        sdnnode[1].port++ <--> Channel_c{datarate=86kbps;} <--> sdnnode[9].port++;
        sdnnode[1].port++ <--> Channel_c{datarate=81kbps;} <--> sdnnode[14].port++;
        sdnnode[1].port++ <--> Channel_c{datarate=83kbps;} <--> sdnnode[21].port++;
        sdnnode[1].port++ <--> Channel_c{datarate=94kbps;} <--> sdnnode[28].port++;
        sdnnode[2].port++ <--> Channel_c{datarate=87kbps;} <--> sdnnode[12].port++;
        sdnnode[2].port++ <--> Channel_c{datarate=76kbps;} <--> sdnnode[15].port++;
        sdnnode[2].port++ <--> Channel_c{datarate=85kbps;} <--> sdnnode[17].port++;
        sdnnode[2].port++ <--> Channel_c{datarate=73kbps;} <--> sdnnode[33].port++;
        sdnnode[3].port++ <--> Channel_c{datarate=81kbps;} <--> sdnnode[14].port++;
        sdnnode[3].port++ <--> Channel_c{datarate=81kbps;} <--> sdnnode[17].port++;
        sdnnode[3].port++ <--> Channel_c{datarate=87kbps;} <--> sdnnode[25].port++;
        sdnnode[3].port++ <--> Channel_c{datarate=76kbps;} <--> sdnnode[27].port++;
        sdnnode[4].port++ <--> Channel_c{datarate=86kbps;} <--> sdnnode[6].port++;
        sdnnode[4].port++ <--> Channel_c{datarate=89kbps;} <--> sdnnode[18].port++;
        sdnnode[4].port++ <--> Channel_c{datarate=82kbps;} <--> sdnnode[24].port++;
        sdnnode[4].port++ <--> Channel_c{datarate=82kbps;} <--> sdnnode[28].port++;
        sdnnode[5].port++ <--> Channel_c{datarate=86kbps;} <--> sdnnode[25].port++;
        sdnnode[5].port++ <--> Channel_c{datarate=79kbps;} <--> sdnnode[27].port++;
        sdnnode[6].port++ <--> Channel_c{datarate=91kbps;} <--> sdnnode[9].port++;
        sdnnode[6].port++ <--> Channel_c{datarate=73kbps;} <--> sdnnode[21].port++;
        sdnnode[6].port++ <--> Channel_c{datarate=89kbps;} <--> sdnnode[24].port++;
        sdnnode[6].port++ <--> Channel_c{datarate=92kbps;} <--> sdnnode[28].port++;
        sdnnode[6].port++ <--> Channel_c{datarate=72kbps;} <--> sdnnode[32].port++;
        sdnnode[7].port++ <--> Channel_c{datarate=80kbps;} <--> sdnnode[9].port++;
        sdnnode[7].port++ <--> Channel_c{datarate=86kbps;} <--> sdnnode[14].port++;
        sdnnode[7].port++ <--> Channel_c{datarate=88kbps;} <--> sdnnode[23].port++;
        sdnnode[7].port++ <--> Channel_c{datarate=73kbps;} <--> sdnnode[31].port++;
        sdnnode[8].port++ <--> Channel_c{datarate=96kbps;} <--> sdnnode[11].port++;
        sdnnode[8].port++ <--> Channel_c{datarate=76kbps;} <--> sdnnode[13].port++;
        sdnnode[8].port++ <--> Channel_c{datarate=75kbps;} <--> sdnnode[19].port++;
        sdnnode[8].port++ <--> Channel_c{datarate=85kbps;} <--> sdnnode[22].port++;
        sdnnode[8].port++ <--> Channel_c{datarate=97kbps;} <--> sdnnode[28].port++;
        sdnnode[9].port++ <--> Channel_c{datarate=74kbps;} <--> sdnnode[11].port++;
        sdnnode[9].port++ <--> Channel_c{datarate=97kbps;} <--> sdnnode[14].port++;
        sdnnode[9].port++ <--> Channel_c{datarate=77kbps;} <--> sdnnode[23].port++;
        sdnnode[9].port++ <--> Channel_c{datarate=83kbps;} <--> sdnnode[28].port++;
        sdnnode[10].port++ <-->Channel_c{datarate=77kbps;} <--> sdnnode[15].port++;
        sdnnode[10].port++ <-->Channel_c{datarate=84kbps;} <--> sdnnode[33].port++;
        sdnnode[11].port++ <-->Channel_c{datarate=85kbps;} <--> sdnnode[22].port++;
        sdnnode[11].port++ <-->Channel_c{datarate=79kbps;} <--> sdnnode[23].port++;
        sdnnode[11].port++ <-->Channel_c{datarate=97kbps;} <--> sdnnode[28].port++;
        sdnnode[12].port++ <-->Channel_c{datarate=87kbps;} <--> sdnnode[16].port++;
        sdnnode[12].port++ <-->Channel_c{datarate=99kbps;} <--> sdnnode[33].port++;
        sdnnode[13].port++ <-->Channel_c{datarate=93kbps;} <--> sdnnode[22].port++;
        sdnnode[14].port++ <-->Channel_c{datarate=80kbps;} <--> sdnnode[25].port++;
        sdnnode[15].port++ <-->Channel_c{datarate=91kbps;} <--> sdnnode[17].port++;
        sdnnode[15].port++ <-->Channel_c{datarate=72kbps;} <--> sdnnode[27].port++;
        sdnnode[16].port++ <-->Channel_c{datarate=71kbps;} <--> sdnnode[21].port++;
        sdnnode[16].port++ <-->Channel_c{datarate=75kbps;} <--> sdnnode[26].port++;
        sdnnode[16].port++ <-->Channel_c{datarate=76kbps;} <--> sdnnode[33].port++;
        sdnnode[16].port++ <-->Channel_c{datarate=98kbps;} <--> sdnnode[34].port++;
        sdnnode[17].port++ <-->Channel_c{datarate=97kbps;} <--> sdnnode[21].port++;
        sdnnode[17].port++ <-->Channel_c{datarate=80kbps;} <--> sdnnode[25].port++;
        sdnnode[17].port++ <-->Channel_c{datarate=71kbps;} <--> sdnnode[26].port++;
        sdnnode[17].port++ <-->Channel_c{datarate=70kbps;} <--> sdnnode[27].port++;
        sdnnode[18].port++ <-->Channel_c{datarate=92kbps;} <--> sdnnode[22].port++;
        sdnnode[18].port++ <-->Channel_c{datarate=75kbps;} <--> sdnnode[28].port++;
        sdnnode[19].port++ <-->Channel_c{datarate=72kbps;} <--> sdnnode[23].port++;
        sdnnode[20].port++ <-->Channel_c{datarate=74kbps;} <--> sdnnode[26].port++;
        sdnnode[20].port++ <-->Channel_c{datarate=89kbps;} <--> sdnnode[29].port++;
        sdnnode[20].port++ <-->Channel_c{datarate=97kbps;} <--> sdnnode[30].port++;
        sdnnode[21].port++ <-->Channel_c{datarate=81kbps;} <--> sdnnode[32].port++;
        sdnnode[21].port++ <-->Channel_c{datarate=74kbps;} <--> sdnnode[34].port++;
        sdnnode[22].port++ <-->Channel_c{datarate=97kbps;} <--> sdnnode[28].port++;
        sdnnode[23].port++ <-->Channel_c{datarate=94kbps;} <--> sdnnode[31].port++;
        sdnnode[24].port++ <-->Channel_c{datarate=80kbps;} <--> sdnnode[32].port++;
        sdnnode[25].port++ <-->Channel_c{datarate=77kbps;} <--> sdnnode[31].port++;
        sdnnode[26].port++ <-->Channel_c{datarate=79kbps;} <--> sdnnode[29].port++;
        sdnnode[26].port++ <-->Channel_c{datarate=89kbps;} <--> sdnnode[34].port++;
        sdnnode[28].port++ <-->Channel_c{datarate=95kbps;} <--> sdnnode[32].port++;
        sdnnode[29].port++ <-->Channel_c{datarate=97kbps;} <--> sdnnode[30].port++;
        sdnnode[29].port++ <-->Channel_c{datarate=95kbps;} <--> sdnnode[32].port++;
        sdnnode[29].port++ <-->Channel_c{datarate=99kbps;} <--> sdnnode[34].port++;
        sdnnode[30].port++ <-->Channel_c{datarate=90kbps;} <--> sdnnode[32].port++;
        sdnnode[32].port++ <-->Channel_c{datarate=94kbps;} <--> sdnnode[34].port++;
      }
