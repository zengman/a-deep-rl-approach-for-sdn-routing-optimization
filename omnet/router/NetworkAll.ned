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
            datarate = 25Mbps;
            ber = uniform(0.0001, 0.005);
            delay = uniform(0.01ms, 0.1ms);
            per = uniform(0.0001,0.005);
            //= this.length./200 000km* 1s;
            double jitter;
            jitter = uniform(0.0001, 0.005);
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
        sdnnode[0].port++ <--> Channel <--> sdnnode[24].port++;
        sdnnode[0].port++ <--> Channel <--> sdnnode[30].port++;
        sdnnode[0].port++ <--> Channel <--> sdnnode[32].port++;
        sdnnode[1].port++ <--> Channel <--> sdnnode[3].port++;
        sdnnode[1].port++ <--> Channel <--> sdnnode[6].port++;
        sdnnode[1].port++ <--> Channel <--> sdnnode[9].port++;
        sdnnode[1].port++ <--> Channel <--> sdnnode[14].port++;
        sdnnode[1].port++ <--> Channel <--> sdnnode[21].port++;
        sdnnode[1].port++ <--> Channel <--> sdnnode[28].port++;
        sdnnode[2].port++ <--> Channel <--> sdnnode[12].port++;
        sdnnode[2].port++ <--> Channel <--> sdnnode[15].port++;
        sdnnode[2].port++ <--> Channel <--> sdnnode[17].port++;
        sdnnode[2].port++ <--> Channel <--> sdnnode[33].port++;
        sdnnode[3].port++ <--> Channel <--> sdnnode[14].port++;
        sdnnode[3].port++ <--> Channel <--> sdnnode[17].port++;
        sdnnode[3].port++ <--> Channel <--> sdnnode[25].port++;
        sdnnode[3].port++ <--> Channel <--> sdnnode[27].port++;
        sdnnode[4].port++ <--> Channel <--> sdnnode[6].port++;
        sdnnode[4].port++ <--> Channel <--> sdnnode[18].port++;
        sdnnode[4].port++ <--> Channel <--> sdnnode[24].port++;
        sdnnode[4].port++ <--> Channel <--> sdnnode[28].port++;
        sdnnode[5].port++ <--> Channel <--> sdnnode[25].port++;
        sdnnode[5].port++ <--> Channel <--> sdnnode[27].port++;
        sdnnode[6].port++ <--> Channel <--> sdnnode[9].port++;
        sdnnode[6].port++ <--> Channel <--> sdnnode[21].port++;
        sdnnode[6].port++ <--> Channel <--> sdnnode[24].port++;
        sdnnode[6].port++ <--> Channel <--> sdnnode[28].port++;
        sdnnode[6].port++ <--> Channel <--> sdnnode[32].port++;
        sdnnode[7].port++ <--> Channel <--> sdnnode[9].port++;
        sdnnode[7].port++ <--> Channel <--> sdnnode[14].port++;
        sdnnode[7].port++ <--> Channel <--> sdnnode[23].port++;
        sdnnode[7].port++ <--> Channel <--> sdnnode[31].port++;
        sdnnode[8].port++ <--> Channel <--> sdnnode[11].port++;
        sdnnode[8].port++ <--> Channel <--> sdnnode[13].port++;
        sdnnode[8].port++ <--> Channel <--> sdnnode[19].port++;
        sdnnode[8].port++ <--> Channel <--> sdnnode[22].port++;
        sdnnode[8].port++ <--> Channel <--> sdnnode[28].port++;
        sdnnode[9].port++ <--> Channel <--> sdnnode[11].port++;
        sdnnode[9].port++ <--> Channel <--> sdnnode[14].port++;
        sdnnode[9].port++ <--> Channel <--> sdnnode[23].port++;
        sdnnode[9].port++ <--> Channel <--> sdnnode[28].port++;
        sdnnode[10].port++ <--> Channel <--> sdnnode[15].port++;
        sdnnode[10].port++ <--> Channel <--> sdnnode[33].port++;
        sdnnode[11].port++ <--> Channel <--> sdnnode[22].port++;
        sdnnode[11].port++ <--> Channel <--> sdnnode[23].port++;
        sdnnode[11].port++ <--> Channel <--> sdnnode[28].port++;
        sdnnode[12].port++ <--> Channel <--> sdnnode[16].port++;
        sdnnode[12].port++ <--> Channel <--> sdnnode[33].port++;
        sdnnode[13].port++ <--> Channel <--> sdnnode[22].port++;
        sdnnode[14].port++ <--> Channel <--> sdnnode[25].port++;
        sdnnode[15].port++ <--> Channel <--> sdnnode[17].port++;
        sdnnode[15].port++ <--> Channel <--> sdnnode[27].port++;
        sdnnode[16].port++ <--> Channel <--> sdnnode[21].port++;
        sdnnode[16].port++ <--> Channel <--> sdnnode[26].port++;
        sdnnode[16].port++ <--> Channel <--> sdnnode[33].port++;
        sdnnode[16].port++ <--> Channel <--> sdnnode[34].port++;
        sdnnode[17].port++ <--> Channel <--> sdnnode[21].port++;
        sdnnode[17].port++ <--> Channel <--> sdnnode[25].port++;
        sdnnode[17].port++ <--> Channel <--> sdnnode[26].port++;
        sdnnode[17].port++ <--> Channel <--> sdnnode[27].port++;
        sdnnode[18].port++ <--> Channel <--> sdnnode[22].port++;
        sdnnode[18].port++ <--> Channel <--> sdnnode[28].port++;
        sdnnode[19].port++ <--> Channel <--> sdnnode[23].port++;
        sdnnode[20].port++ <--> Channel <--> sdnnode[26].port++;
        sdnnode[20].port++ <--> Channel <--> sdnnode[29].port++;
        sdnnode[20].port++ <--> Channel <--> sdnnode[30].port++;
        sdnnode[21].port++ <--> Channel <--> sdnnode[32].port++;
        sdnnode[21].port++ <--> Channel <--> sdnnode[34].port++;
        sdnnode[22].port++ <--> Channel <--> sdnnode[28].port++;
        sdnnode[23].port++ <--> Channel <--> sdnnode[31].port++;
        sdnnode[24].port++ <--> Channel <--> sdnnode[32].port++;
        sdnnode[25].port++ <--> Channel <--> sdnnode[31].port++;
        sdnnode[26].port++ <--> Channel <--> sdnnode[29].port++;
        sdnnode[26].port++ <--> Channel <--> sdnnode[34].port++;
        sdnnode[28].port++ <--> Channel <--> sdnnode[32].port++;
        sdnnode[29].port++ <--> Channel <--> sdnnode[30].port++;
        sdnnode[29].port++ <--> Channel <--> sdnnode[32].port++;
        sdnnode[29].port++ <--> Channel <--> sdnnode[34].port++;
        sdnnode[30].port++ <--> Channel <--> sdnnode[32].port++;
        sdnnode[32].port++ <--> Channel <--> sdnnode[34].port++;
        
        
        
        
        
        
        
        
        
      }
