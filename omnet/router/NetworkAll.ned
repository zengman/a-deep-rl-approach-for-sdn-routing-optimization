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
        sdnnode[0].port++ <--> Channel_c{datarate=31kbps;} <--> sdnnode[24].port++;
        sdnnode[0].port++ <--> Channel_c{datarate=31kbps;} <--> sdnnode[30].port++;
        sdnnode[0].port++ <--> Channel_c{datarate=31kbps;} <--> sdnnode[32].port++;
        sdnnode[1].port++ <--> Channel_c{datarate=34kbps;} <--> sdnnode[3].port++;
        sdnnode[1].port++ <--> Channel_c{datarate=24kbps;} <--> sdnnode[6].port++;
        sdnnode[1].port++ <--> Channel_c{datarate=33kbps;} <--> sdnnode[9].port++;
        sdnnode[1].port++ <--> Channel_c{datarate=33kbps;} <--> sdnnode[14].port++;
        sdnnode[1].port++ <--> Channel_c{datarate=34kbps;} <--> sdnnode[21].port++;
        sdnnode[1].port++ <--> Channel_c{datarate=40kbps;} <--> sdnnode[28].port++;
        sdnnode[2].port++ <--> Channel_c{datarate=37kbps;} <--> sdnnode[12].port++;
        sdnnode[2].port++ <--> Channel_c{datarate=24kbps;} <--> sdnnode[15].port++;
        sdnnode[2].port++ <--> Channel_c{datarate=36kbps;} <--> sdnnode[17].port++;
        sdnnode[2].port++ <--> Channel_c{datarate=26kbps;} <--> sdnnode[33].port++;
        sdnnode[3].port++ <--> Channel_c{datarate=20kbps;} <--> sdnnode[14].port++;
        sdnnode[3].port++ <--> Channel_c{datarate=31kbps;} <--> sdnnode[17].port++;
        sdnnode[3].port++ <--> Channel_c{datarate=34kbps;} <--> sdnnode[25].port++;
        sdnnode[3].port++ <--> Channel_c{datarate=29kbps;} <--> sdnnode[27].port++;
        sdnnode[4].port++ <--> Channel_c{datarate=34kbps;} <--> sdnnode[6].port++;
        sdnnode[4].port++ <--> Channel_c{datarate=29kbps;} <--> sdnnode[18].port++;
        sdnnode[4].port++ <--> Channel_c{datarate=38kbps;} <--> sdnnode[24].port++;
        sdnnode[4].port++ <--> Channel_c{datarate=40kbps;} <--> sdnnode[28].port++;
        sdnnode[5].port++ <--> Channel_c{datarate=40kbps;} <--> sdnnode[25].port++;
        sdnnode[5].port++ <--> Channel_c{datarate=36kbps;} <--> sdnnode[27].port++;
        sdnnode[6].port++ <--> Channel_c{datarate=23kbps;} <--> sdnnode[9].port++;
        sdnnode[6].port++ <--> Channel_c{datarate=40kbps;} <--> sdnnode[21].port++;
        sdnnode[6].port++ <--> Channel_c{datarate=20kbps;} <--> sdnnode[24].port++;
        sdnnode[6].port++ <--> Channel_c{datarate=23kbps;} <--> sdnnode[28].port++;
        sdnnode[6].port++ <--> Channel_c{datarate=21kbps;} <--> sdnnode[32].port++;
        sdnnode[7].port++ <--> Channel_c{datarate=30kbps;} <--> sdnnode[9].port++;
        sdnnode[7].port++ <--> Channel_c{datarate=31kbps;} <--> sdnnode[14].port++;
        sdnnode[7].port++ <--> Channel_c{datarate=24kbps;} <--> sdnnode[23].port++;
        sdnnode[7].port++ <--> Channel_c{datarate=29kbps;} <--> sdnnode[31].port++;
        sdnnode[8].port++ <--> Channel_c{datarate=24kbps;} <--> sdnnode[11].port++;
        sdnnode[8].port++ <--> Channel_c{datarate=38kbps;} <--> sdnnode[13].port++;
        sdnnode[8].port++ <--> Channel_c{datarate=33kbps;} <--> sdnnode[19].port++;
        sdnnode[8].port++ <--> Channel_c{datarate=37kbps;} <--> sdnnode[22].port++;
        sdnnode[8].port++ <--> Channel_c{datarate=32kbps;} <--> sdnnode[28].port++;
        sdnnode[9].port++ <--> Channel_c{datarate=35kbps;} <--> sdnnode[11].port++;
        sdnnode[9].port++ <--> Channel_c{datarate=31kbps;} <--> sdnnode[14].port++;
        sdnnode[9].port++ <--> Channel_c{datarate=40kbps;} <--> sdnnode[23].port++;
        sdnnode[9].port++ <--> Channel_c{datarate=24kbps;} <--> sdnnode[28].port++;
        sdnnode[10].port++ <-->Channel_c{datarate=26kbps;} <--> sdnnode[15].port++;
        sdnnode[10].port++ <-->Channel_c{datarate=21kbps;} <--> sdnnode[33].port++;
        sdnnode[11].port++ <-->Channel_c{datarate=30kbps;} <--> sdnnode[22].port++;
        sdnnode[11].port++ <-->Channel_c{datarate=20kbps;} <--> sdnnode[23].port++;
        sdnnode[11].port++ <-->Channel_c{datarate=21kbps;} <--> sdnnode[28].port++;
        sdnnode[12].port++ <-->Channel_c{datarate=27kbps;} <--> sdnnode[16].port++;
        sdnnode[12].port++ <-->Channel_c{datarate=37kbps;} <--> sdnnode[33].port++;
        sdnnode[13].port++ <-->Channel_c{datarate=24kbps;} <--> sdnnode[22].port++;
        sdnnode[14].port++ <-->Channel_c{datarate=20kbps;} <--> sdnnode[25].port++;
        sdnnode[15].port++ <-->Channel_c{datarate=32kbps;} <--> sdnnode[17].port++;
        sdnnode[15].port++ <-->Channel_c{datarate=36kbps;} <--> sdnnode[27].port++;
        sdnnode[16].port++ <-->Channel_c{datarate=30kbps;} <--> sdnnode[21].port++;
        sdnnode[16].port++ <-->Channel_c{datarate=32kbps;} <--> sdnnode[26].port++;
        sdnnode[16].port++ <-->Channel_c{datarate=40kbps;} <--> sdnnode[33].port++;
        sdnnode[16].port++ <-->Channel_c{datarate=35kbps;} <--> sdnnode[34].port++;
        sdnnode[17].port++ <-->Channel_c{datarate=21kbps;} <--> sdnnode[21].port++;
        sdnnode[17].port++ <-->Channel_c{datarate=26kbps;} <--> sdnnode[25].port++;
        sdnnode[17].port++ <-->Channel_c{datarate=35kbps;} <--> sdnnode[26].port++;
        sdnnode[17].port++ <-->Channel_c{datarate=22kbps;} <--> sdnnode[27].port++;
        sdnnode[18].port++ <-->Channel_c{datarate=23kbps;} <--> sdnnode[22].port++;
        sdnnode[18].port++ <-->Channel_c{datarate=36kbps;} <--> sdnnode[28].port++;
        sdnnode[19].port++ <-->Channel_c{datarate=26kbps;} <--> sdnnode[23].port++;
        sdnnode[20].port++ <-->Channel_c{datarate=28kbps;} <--> sdnnode[26].port++;
        sdnnode[20].port++ <-->Channel_c{datarate=33kbps;} <--> sdnnode[29].port++;
        sdnnode[20].port++ <-->Channel_c{datarate=39kbps;} <--> sdnnode[30].port++;
        sdnnode[21].port++ <-->Channel_c{datarate=31kbps;} <--> sdnnode[32].port++;
        sdnnode[21].port++ <-->Channel_c{datarate=37kbps;} <--> sdnnode[34].port++;
        sdnnode[22].port++ <-->Channel_c{datarate=32kbps;} <--> sdnnode[28].port++;
        sdnnode[23].port++ <-->Channel_c{datarate=20kbps;} <--> sdnnode[31].port++;
        sdnnode[24].port++ <-->Channel_c{datarate=32kbps;} <--> sdnnode[32].port++;
        sdnnode[25].port++ <-->Channel_c{datarate=39kbps;} <--> sdnnode[31].port++;
        sdnnode[26].port++ <-->Channel_c{datarate=35kbps;} <--> sdnnode[29].port++;
        sdnnode[26].port++ <-->Channel_c{datarate=26kbps;} <--> sdnnode[34].port++;
        sdnnode[28].port++ <-->Channel_c{datarate=21kbps;} <--> sdnnode[32].port++;
        sdnnode[29].port++ <-->Channel_c{datarate=27kbps;} <--> sdnnode[30].port++;
        sdnnode[29].port++ <-->Channel_c{datarate=36kbps;} <--> sdnnode[32].port++;
        sdnnode[29].port++ <-->Channel_c{datarate=36kbps;} <--> sdnnode[34].port++;
        sdnnode[30].port++ <-->Channel_c{datarate=37kbps;} <--> sdnnode[32].port++;
        sdnnode[32].port++ <-->Channel_c{datarate=24kbps;} <--> sdnnode[34].port++;
      }
