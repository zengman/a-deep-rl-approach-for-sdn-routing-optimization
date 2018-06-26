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
        sdnnode[0].port++ <--> Channel_c{datarate=14.0kbps;} <--> sdnnode[24].port++;
        sdnnode[0].port++ <--> Channel_c{datarate=16.0kbps;} <--> sdnnode[30].port++;
        sdnnode[0].port++ <--> Channel_c{datarate=16.6kbps;} <--> sdnnode[32].port++;
        sdnnode[1].port++ <--> Channel_c{datarate=16.0kbps;} <--> sdnnode[3].port++;
        sdnnode[1].port++ <--> Channel_c{datarate=15.200000000000001kbps;} <--> sdnnode[6].port++;
        sdnnode[1].port++ <--> Channel_c{datarate=18.400000000000002kbps;} <--> sdnnode[9].port++;
        sdnnode[1].port++ <--> Channel_c{datarate=15.200000000000001kbps;} <--> sdnnode[14].port++;
        sdnnode[1].port++ <--> Channel_c{datarate=18.6kbps;} <--> sdnnode[21].port++;
        sdnnode[1].port++ <--> Channel_c{datarate=17.400000000000002kbps;} <--> sdnnode[28].port++;
        sdnnode[2].port++ <--> Channel_c{datarate=17.6kbps;} <--> sdnnode[12].port++;
        sdnnode[2].port++ <--> Channel_c{datarate=17.6kbps;} <--> sdnnode[15].port++;
        sdnnode[2].port++ <--> Channel_c{datarate=14.600000000000001kbps;} <--> sdnnode[17].port++;
        sdnnode[2].port++ <--> Channel_c{datarate=17.6kbps;} <--> sdnnode[33].port++;
        sdnnode[3].port++ <--> Channel_c{datarate=18.400000000000002kbps;} <--> sdnnode[14].port++;
        sdnnode[3].port++ <--> Channel_c{datarate=17.0kbps;} <--> sdnnode[17].port++;
        sdnnode[3].port++ <--> Channel_c{datarate=18.2kbps;} <--> sdnnode[25].port++;
        sdnnode[3].port++ <--> Channel_c{datarate=17.0kbps;} <--> sdnnode[27].port++;
        sdnnode[4].port++ <--> Channel_c{datarate=18.0kbps;} <--> sdnnode[6].port++;
        sdnnode[4].port++ <--> Channel_c{datarate=16.6kbps;} <--> sdnnode[18].port++;
        sdnnode[4].port++ <--> Channel_c{datarate=18.8kbps;} <--> sdnnode[24].port++;
        sdnnode[4].port++ <--> Channel_c{datarate=18.6kbps;} <--> sdnnode[28].port++;
        sdnnode[5].port++ <--> Channel_c{datarate=15.4kbps;} <--> sdnnode[25].port++;
        sdnnode[5].port++ <--> Channel_c{datarate=19.0kbps;} <--> sdnnode[27].port++;
        sdnnode[6].port++ <--> Channel_c{datarate=16.400000000000002kbps;} <--> sdnnode[9].port++;
        sdnnode[6].port++ <--> Channel_c{datarate=14.0kbps;} <--> sdnnode[21].port++;
        sdnnode[6].port++ <--> Channel_c{datarate=15.8kbps;} <--> sdnnode[24].port++;
        sdnnode[6].port++ <--> Channel_c{datarate=15.4kbps;} <--> sdnnode[28].port++;
        sdnnode[6].port++ <--> Channel_c{datarate=15.200000000000001kbps;} <--> sdnnode[32].port++;
        sdnnode[7].port++ <--> Channel_c{datarate=15.4kbps;} <--> sdnnode[9].port++;
        sdnnode[7].port++ <--> Channel_c{datarate=14.8kbps;} <--> sdnnode[14].port++;
        sdnnode[7].port++ <--> Channel_c{datarate=15.200000000000001kbps;} <--> sdnnode[23].port++;
        sdnnode[7].port++ <--> Channel_c{datarate=18.6kbps;} <--> sdnnode[31].port++;
        sdnnode[8].port++ <--> Channel_c{datarate=19.400000000000002kbps;} <--> sdnnode[11].port++;
        sdnnode[8].port++ <--> Channel_c{datarate=15.8kbps;} <--> sdnnode[13].port++;
        sdnnode[8].port++ <--> Channel_c{datarate=17.2kbps;} <--> sdnnode[19].port++;
        sdnnode[8].port++ <--> Channel_c{datarate=17.400000000000002kbps;} <--> sdnnode[22].port++;
        sdnnode[8].port++ <--> Channel_c{datarate=15.4kbps;} <--> sdnnode[28].port++;
        sdnnode[9].port++ <--> Channel_c{datarate=14.0kbps;} <--> sdnnode[11].port++;
        sdnnode[9].port++ <--> Channel_c{datarate=17.2kbps;} <--> sdnnode[14].port++;
        sdnnode[9].port++ <--> Channel_c{datarate=18.400000000000002kbps;} <--> sdnnode[23].port++;
        sdnnode[9].port++ <--> Channel_c{datarate=16.8kbps;} <--> sdnnode[28].port++;
        sdnnode[10].port++ <-->Channel_c{datarate=16.6kbps;} <--> sdnnode[15].port++;
        sdnnode[10].port++ <-->Channel_c{datarate=19.8kbps;} <--> sdnnode[33].port++;
        sdnnode[11].port++ <-->Channel_c{datarate=18.2kbps;} <--> sdnnode[22].port++;
        sdnnode[11].port++ <-->Channel_c{datarate=18.0kbps;} <--> sdnnode[23].port++;
        sdnnode[11].port++ <-->Channel_c{datarate=14.0kbps;} <--> sdnnode[28].port++;
        sdnnode[12].port++ <-->Channel_c{datarate=19.0kbps;} <--> sdnnode[16].port++;
        sdnnode[12].port++ <-->Channel_c{datarate=16.8kbps;} <--> sdnnode[33].port++;
        sdnnode[13].port++ <-->Channel_c{datarate=14.600000000000001kbps;} <--> sdnnode[22].port++;
        sdnnode[14].port++ <-->Channel_c{datarate=17.0kbps;} <--> sdnnode[25].port++;
        sdnnode[15].port++ <-->Channel_c{datarate=14.4kbps;} <--> sdnnode[17].port++;
        sdnnode[15].port++ <-->Channel_c{datarate=15.8kbps;} <--> sdnnode[27].port++;
        sdnnode[16].port++ <-->Channel_c{datarate=17.400000000000002kbps;} <--> sdnnode[21].port++;
        sdnnode[16].port++ <-->Channel_c{datarate=18.0kbps;} <--> sdnnode[26].port++;
        sdnnode[16].port++ <-->Channel_c{datarate=18.400000000000002kbps;} <--> sdnnode[33].port++;
        sdnnode[16].port++ <-->Channel_c{datarate=14.600000000000001kbps;} <--> sdnnode[34].port++;
        sdnnode[17].port++ <-->Channel_c{datarate=14.0kbps;} <--> sdnnode[21].port++;
        sdnnode[17].port++ <-->Channel_c{datarate=19.8kbps;} <--> sdnnode[25].port++;
        sdnnode[17].port++ <-->Channel_c{datarate=17.0kbps;} <--> sdnnode[26].port++;
        sdnnode[17].port++ <-->Channel_c{datarate=17.400000000000002kbps;} <--> sdnnode[27].port++;
        sdnnode[18].port++ <-->Channel_c{datarate=17.6kbps;} <--> sdnnode[22].port++;
        sdnnode[18].port++ <-->Channel_c{datarate=16.6kbps;} <--> sdnnode[28].port++;
        sdnnode[19].port++ <-->Channel_c{datarate=17.2kbps;} <--> sdnnode[23].port++;
        sdnnode[20].port++ <-->Channel_c{datarate=15.4kbps;} <--> sdnnode[26].port++;
        sdnnode[20].port++ <-->Channel_c{datarate=14.8kbps;} <--> sdnnode[29].port++;
        sdnnode[20].port++ <-->Channel_c{datarate=16.8kbps;} <--> sdnnode[30].port++;
        sdnnode[21].port++ <-->Channel_c{datarate=15.600000000000001kbps;} <--> sdnnode[32].port++;
        sdnnode[21].port++ <-->Channel_c{datarate=16.0kbps;} <--> sdnnode[34].port++;
        sdnnode[22].port++ <-->Channel_c{datarate=18.2kbps;} <--> sdnnode[28].port++;
        sdnnode[23].port++ <-->Channel_c{datarate=15.200000000000001kbps;} <--> sdnnode[31].port++;
        sdnnode[24].port++ <-->Channel_c{datarate=14.600000000000001kbps;} <--> sdnnode[32].port++;
        sdnnode[25].port++ <-->Channel_c{datarate=19.200000000000003kbps;} <--> sdnnode[31].port++;
        sdnnode[26].port++ <-->Channel_c{datarate=19.400000000000002kbps;} <--> sdnnode[29].port++;
        sdnnode[26].port++ <-->Channel_c{datarate=16.400000000000002kbps;} <--> sdnnode[34].port++;
        sdnnode[28].port++ <-->Channel_c{datarate=19.0kbps;} <--> sdnnode[32].port++;
        sdnnode[29].port++ <-->Channel_c{datarate=18.2kbps;} <--> sdnnode[30].port++;
        sdnnode[29].port++ <-->Channel_c{datarate=16.8kbps;} <--> sdnnode[32].port++;
        sdnnode[29].port++ <-->Channel_c{datarate=15.200000000000001kbps;} <--> sdnnode[34].port++;
        sdnnode[30].port++ <-->Channel_c{datarate=16.0kbps;} <--> sdnnode[32].port++;
        sdnnode[32].port++ <-->Channel_c{datarate=16.2kbps;} <--> sdnnode[34].port++;
      }
