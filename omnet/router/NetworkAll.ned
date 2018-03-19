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
            //datarate = 25Mbps;
            //ber = uniform(0.0001, 0.005);
            //delay = uniform(0.01ms, 0.1ms);
            //per = uniform(0.0001,0.005);
            //= this.length./200 000km* 1s;
            double jitter;
            jitter = uniform(0.0001, 0.005);
            delay = 0.2ms;
        }
        
        channel Channel_c extends ned.DatarateChannel
        {
            delay = 0.2ms;
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
        sdnnode[0].port++ <--> Channel_c{datarate=31Mbps;} <--> sdnnode[24].port++;
        sdnnode[0].port++ <--> Channel_c{datarate=31Mbps;} <--> sdnnode[30].port++;
        sdnnode[0].port++ <--> Channel_c{datarate=31Mbps;} <--> sdnnode[32].port++;
        sdnnode[1].port++ <--> Channel_c{datarate=34Mbps;} <--> sdnnode[3].port++;
        sdnnode[1].port++ <--> Channel_c{datarate=24Mbps;} <--> sdnnode[6].port++;
        sdnnode[1].port++ <--> Channel_c{datarate=33Mbps;} <--> sdnnode[9].port++;
        sdnnode[1].port++ <--> Channel_c{datarate=33Mbps;} <--> sdnnode[14].port++;
        sdnnode[1].port++ <--> Channel_c{datarate=34Mbps;} <--> sdnnode[21].port++;
        sdnnode[1].port++ <--> Channel_c{datarate=40Mbps;} <--> sdnnode[28].port++;
        sdnnode[2].port++ <--> Channel_c{datarate=37Mbps;} <--> sdnnode[12].port++;
        sdnnode[2].port++ <--> Channel_c{datarate=24Mbps;} <--> sdnnode[15].port++;
        sdnnode[2].port++ <--> Channel_c{datarate=36Mbps;} <--> sdnnode[17].port++;
        sdnnode[2].port++ <--> Channel_c{datarate=26Mbps;} <--> sdnnode[33].port++;
        sdnnode[3].port++ <--> Channel_c{datarate=20Mbps;} <--> sdnnode[14].port++;
        sdnnode[3].port++ <--> Channel_c{datarate=31Mbps;} <--> sdnnode[17].port++;
        sdnnode[3].port++ <--> Channel_c{datarate=34Mbps;} <--> sdnnode[25].port++;
        sdnnode[3].port++ <--> Channel_c{datarate=29Mbps;} <--> sdnnode[27].port++;
        sdnnode[4].port++ <--> Channel_c{datarate=34Mbps;} <--> sdnnode[6].port++;
        sdnnode[4].port++ <--> Channel_c{datarate=29Mbps;} <--> sdnnode[18].port++;
        sdnnode[4].port++ <--> Channel_c{datarate=38Mbps;} <--> sdnnode[24].port++;
        sdnnode[4].port++ <--> Channel_c{datarate=40Mbps;} <--> sdnnode[28].port++;
        sdnnode[5].port++ <--> Channel_c{datarate=40Mbps;} <--> sdnnode[25].port++;
        sdnnode[5].port++ <--> Channel_c{datarate=36Mbps;} <--> sdnnode[27].port++;
        sdnnode[6].port++ <--> Channel_c{datarate=23Mbps;} <--> sdnnode[9].port++;
        sdnnode[6].port++ <--> Channel_c{datarate=40Mbps;} <--> sdnnode[21].port++;
        sdnnode[6].port++ <--> Channel_c{datarate=20Mbps;} <--> sdnnode[24].port++;
        sdnnode[6].port++ <--> Channel_c{datarate=23Mbps;} <--> sdnnode[28].port++;
        sdnnode[6].port++ <--> Channel_c{datarate=21Mbps;} <--> sdnnode[32].port++;
        sdnnode[7].port++ <--> Channel_c{datarate=30Mbps;} <--> sdnnode[9].port++;
        sdnnode[7].port++ <--> Channel_c{datarate=31Mbps;} <--> sdnnode[14].port++;
        sdnnode[7].port++ <--> Channel_c{datarate=24Mbps;} <--> sdnnode[23].port++;
        sdnnode[7].port++ <--> Channel_c{datarate=29Mbps;} <--> sdnnode[31].port++;
        sdnnode[8].port++ <--> Channel_c{datarate=24Mbps;} <--> sdnnode[11].port++;
        sdnnode[8].port++ <--> Channel_c{datarate=38Mbps;} <--> sdnnode[13].port++;
        sdnnode[8].port++ <--> Channel_c{datarate=33Mbps;} <--> sdnnode[19].port++;
        sdnnode[8].port++ <--> Channel_c{datarate=37Mbps;} <--> sdnnode[22].port++;
        sdnnode[8].port++ <--> Channel_c{datarate=32Mbps;} <--> sdnnode[28].port++;
        sdnnode[9].port++ <--> Channel_c{datarate=35Mbps;} <--> sdnnode[11].port++;
        sdnnode[9].port++ <--> Channel_c{datarate=31Mbps;} <--> sdnnode[14].port++;
        sdnnode[9].port++ <--> Channel_c{datarate=40Mbps;} <--> sdnnode[23].port++;
        sdnnode[9].port++ <--> Channel_c{datarate=24Mbps;} <--> sdnnode[28].port++;
        sdnnode[10].port++ <-->Channel_c{datarate=26Mbps;} <--> sdnnode[15].port++;
        sdnnode[10].port++ <-->Channel_c{datarate=21Mbps;} <--> sdnnode[33].port++;
        sdnnode[11].port++ <-->Channel_c{datarate=30Mbps;} <--> sdnnode[22].port++;
        sdnnode[11].port++ <-->Channel_c{datarate=20Mbps;} <--> sdnnode[23].port++;
        sdnnode[11].port++ <-->Channel_c{datarate=21Mbps;} <--> sdnnode[28].port++;
        sdnnode[12].port++ <-->Channel_c{datarate=27Mbps;} <--> sdnnode[16].port++;
        sdnnode[12].port++ <-->Channel_c{datarate=37Mbps;} <--> sdnnode[33].port++;
        sdnnode[13].port++ <-->Channel_c{datarate=24Mbps;} <--> sdnnode[22].port++;
        sdnnode[14].port++ <-->Channel_c{datarate=20Mbps;} <--> sdnnode[25].port++;
        sdnnode[15].port++ <-->Channel_c{datarate=32Mbps;} <--> sdnnode[17].port++;
        sdnnode[15].port++ <-->Channel_c{datarate=36Mbps;} <--> sdnnode[27].port++;
        sdnnode[16].port++ <-->Channel_c{datarate=30Mbps;} <--> sdnnode[21].port++;
        sdnnode[16].port++ <-->Channel_c{datarate=32Mbps;} <--> sdnnode[26].port++;
        sdnnode[16].port++ <-->Channel_c{datarate=40Mbps;} <--> sdnnode[33].port++;
        sdnnode[16].port++ <-->Channel_c{datarate=35Mbps;} <--> sdnnode[34].port++;
        sdnnode[17].port++ <-->Channel_c{datarate=21Mbps;} <--> sdnnode[21].port++;
        sdnnode[17].port++ <-->Channel_c{datarate=26Mbps;} <--> sdnnode[25].port++;
        sdnnode[17].port++ <-->Channel_c{datarate=35Mbps;} <--> sdnnode[26].port++;
        sdnnode[17].port++ <-->Channel_c{datarate=22Mbps;} <--> sdnnode[27].port++;
        sdnnode[18].port++ <-->Channel_c{datarate=23Mbps;} <--> sdnnode[22].port++;
        sdnnode[18].port++ <-->Channel_c{datarate=36Mbps;} <--> sdnnode[28].port++;
        sdnnode[19].port++ <-->Channel_c{datarate=26Mbps;} <--> sdnnode[23].port++;
        sdnnode[20].port++ <-->Channel_c{datarate=28Mbps;} <--> sdnnode[26].port++;
        sdnnode[20].port++ <-->Channel_c{datarate=33Mbps;} <--> sdnnode[29].port++;
        sdnnode[20].port++ <-->Channel_c{datarate=39Mbps;} <--> sdnnode[30].port++;
        sdnnode[21].port++ <-->Channel_c{datarate=31Mbps;} <--> sdnnode[32].port++;
        sdnnode[21].port++ <-->Channel_c{datarate=37Mbps;} <--> sdnnode[34].port++;
        sdnnode[22].port++ <-->Channel_c{datarate=32Mbps;} <--> sdnnode[28].port++;
        sdnnode[23].port++ <-->Channel_c{datarate=20Mbps;} <--> sdnnode[31].port++;
        sdnnode[24].port++ <-->Channel_c{datarate=32Mbps;} <--> sdnnode[32].port++;
        sdnnode[25].port++ <-->Channel_c{datarate=39Mbps;} <--> sdnnode[31].port++;
        sdnnode[26].port++ <-->Channel_c{datarate=35Mbps;} <--> sdnnode[29].port++;
        sdnnode[26].port++ <-->Channel_c{datarate=26Mbps;} <--> sdnnode[34].port++;
        sdnnode[28].port++ <-->Channel_c{datarate=21Mbps;} <--> sdnnode[32].port++;
        sdnnode[29].port++ <-->Channel_c{datarate=27Mbps;} <--> sdnnode[30].port++;
        sdnnode[29].port++ <-->Channel_c{datarate=36Mbps;} <--> sdnnode[32].port++;
        sdnnode[29].port++ <-->Channel_c{datarate=36Mbps;} <--> sdnnode[34].port++;
        sdnnode[30].port++ <-->Channel_c{datarate=37Mbps;} <--> sdnnode[32].port++;
        sdnnode[32].port++ <-->Channel_c{datarate=24Mbps;} <--> sdnnode[34].port++;
        
        
        
        
        
        
        
        
        
      }
