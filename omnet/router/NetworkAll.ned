package networksimulator;

network NetworkAll
{
    parameters:
        int numNodes = 50;
        int numTx = 50;

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
                    //port[3];
            }

    connections:
       sdnnode[0].port++ <--> Channel_c{datarate=64Mbps;} <--> sdnnode[1].port++;
        sdnnode[0].port++ <--> Channel_c{datarate=71Mbps;} <--> sdnnode[3].port++;
        sdnnode[1].port++ <--> Channel_c{datarate=62Mbps;} <--> sdnnode[2].port++;
        sdnnode[1].port++ <--> Channel_c{datarate=69Mbps;} <--> sdnnode[4].port++;
        sdnnode[2].port++ <--> Channel_c{datarate=75Mbps;} <--> sdnnode[4].port++;
        sdnnode[2].port++ <--> Channel_c{datarate=77Mbps;} <--> sdnnode[8].port++;
        sdnnode[2].port++ <--> Channel_c{datarate=74Mbps;} <--> sdnnode[10].port++;
        sdnnode[3].port++ <--> Channel_c{datarate=70Mbps;} <--> sdnnode[49].port++;
        sdnnode[4].port++ <--> Channel_c{datarate=61Mbps;} <--> sdnnode[5].port++;
        sdnnode[4].port++ <--> Channel_c{datarate=80Mbps;} <--> sdnnode[11].port++;
        sdnnode[4].port++ <--> Channel_c{datarate=77Mbps;} <--> sdnnode[12].port++;
        sdnnode[5].port++ <--> Channel_c{datarate=63Mbps;} <--> sdnnode[12].port++;
        sdnnode[6].port++ <--> Channel_c{datarate=76Mbps;} <--> sdnnode[7].port++;
        sdnnode[6].port++ <--> Channel_c{datarate=80Mbps;} <--> sdnnode[17].port++;
        sdnnode[7].port++ <--> Channel_c{datarate=67Mbps;} <--> sdnnode[9].port++;
        sdnnode[7].port++ <--> Channel_c{datarate=73Mbps;} <--> sdnnode[17].port++;
        sdnnode[8].port++ <--> Channel_c{datarate=74Mbps;} <--> sdnnode[9].port++;
        sdnnode[8].port++ <--> Channel_c{datarate=78Mbps;} <--> sdnnode[10].port++;
        sdnnode[8].port++ <--> Channel_c{datarate=75Mbps;} <--> sdnnode[14].port++;
        sdnnode[8].port++ <--> Channel_c{datarate=62Mbps;} <--> sdnnode[49].port++;
        sdnnode[9].port++ <--> Channel_c{datarate=76Mbps;} <--> sdnnode[13].port++;
        sdnnode[10].port++ <--> Channel_c{datarate=71Mbps;} <--> sdnnode[11].port++;
        sdnnode[10].port++ <--> Channel_c{datarate=70Mbps;} <--> sdnnode[14].port++;
        sdnnode[10].port++ <--> Channel_c{datarate=61Mbps;} <--> sdnnode[22].port++;
        sdnnode[11].port++ <--> Channel_c{datarate=75Mbps;} <--> sdnnode[12].port++;
        sdnnode[11].port++ <--> Channel_c{datarate=79Mbps;} <--> sdnnode[46].port++;
        sdnnode[12].port++ <--> Channel_c{datarate=71Mbps;} <--> sdnnode[46].port++;
        sdnnode[12].port++ <--> Channel_c{datarate=63Mbps;} <--> sdnnode[48].port++;
        sdnnode[13].port++ <--> Channel_c{datarate=78Mbps;} <--> sdnnode[14].port++;
        sdnnode[13].port++ <--> Channel_c{datarate=67Mbps;} <--> sdnnode[15].port++;
        sdnnode[14].port++ <--> Channel_c{datarate=64Mbps;} <--> sdnnode[21].port++;
        sdnnode[15].port++ <--> Channel_c{datarate=80Mbps;} <--> sdnnode[16].port++;
        sdnnode[15].port++ <--> Channel_c{datarate=76Mbps;} <--> sdnnode[21].port++;
        sdnnode[15].port++ <--> Channel_c{datarate=61Mbps;} <--> sdnnode[22].port++;
        sdnnode[16].port++ <--> Channel_c{datarate=80Mbps;} <--> sdnnode[17].port++;
        sdnnode[16].port++ <--> Channel_c{datarate=68Mbps;} <--> sdnnode[18].port++;
        sdnnode[17].port++ <--> Channel_c{datarate=80Mbps;} <--> sdnnode[19].port++;
        sdnnode[18].port++ <--> Channel_c{datarate=71Mbps;} <--> sdnnode[20].port++;
        sdnnode[19].port++ <--> Channel_c{datarate=66Mbps;} <--> sdnnode[20].port++;
        sdnnode[19].port++ <--> Channel_c{datarate=67Mbps;} <--> sdnnode[24].port++;
        sdnnode[20].port++ <--> Channel_c{datarate=78Mbps;} <--> sdnnode[23].port++;
        sdnnode[21].port++ <--> Channel_c{datarate=68Mbps;} <--> sdnnode[23].port++;
        sdnnode[21].port++ <--> Channel_c{datarate=67Mbps;} <--> sdnnode[25].port++;
        sdnnode[22].port++ <--> Channel_c{datarate=62Mbps;} <--> sdnnode[25].port++;
        sdnnode[22].port++ <--> Channel_c{datarate=70Mbps;} <--> sdnnode[26].port++;
        sdnnode[22].port++ <--> Channel_c{datarate=66Mbps;} <--> sdnnode[45].port++;
        sdnnode[23].port++ <--> Channel_c{datarate=65Mbps;} <--> sdnnode[24].port++;
        sdnnode[23].port++ <--> Channel_c{datarate=66Mbps;} <--> sdnnode[27].port++;
        sdnnode[23].port++ <--> Channel_c{datarate=65Mbps;} <--> sdnnode[30].port++;
        sdnnode[24].port++ <--> Channel_c{datarate=68Mbps;} <--> sdnnode[31].port++;
        sdnnode[25].port++ <--> Channel_c{datarate=60Mbps;} <--> sdnnode[26].port++;
        sdnnode[25].port++ <--> Channel_c{datarate=68Mbps;} <--> sdnnode[27].port++;
        sdnnode[26].port++ <--> Channel_c{datarate=70Mbps;} <--> sdnnode[27].port++;
        sdnnode[26].port++ <--> Channel_c{datarate=78Mbps;} <--> sdnnode[44].port++;
        sdnnode[27].port++ <--> Channel_c{datarate=75Mbps;} <--> sdnnode[28].port++;
        sdnnode[28].port++ <--> Channel_c{datarate=67Mbps;} <--> sdnnode[29].port++;
        sdnnode[28].port++ <--> Channel_c{datarate=76Mbps;} <--> sdnnode[30].port++;
        sdnnode[29].port++ <--> Channel_c{datarate=70Mbps;} <--> sdnnode[34].port++;
        sdnnode[30].port++ <--> Channel_c{datarate=75Mbps;} <--> sdnnode[31].port++;
        sdnnode[30].port++ <--> Channel_c{datarate=76Mbps;} <--> sdnnode[34].port++;
        sdnnode[31].port++ <--> Channel_c{datarate=62Mbps;} <--> sdnnode[34].port++;
        sdnnode[32].port++ <--> Channel_c{datarate=78Mbps;} <--> sdnnode[34].port++;
        sdnnode[32].port++ <--> Channel_c{datarate=72Mbps;} <--> sdnnode[35].port++;
        sdnnode[33].port++ <--> Channel_c{datarate=73Mbps;} <--> sdnnode[34].port++;
        sdnnode[33].port++ <--> Channel_c{datarate=65Mbps;} <--> sdnnode[35].port++;
        sdnnode[33].port++ <--> Channel_c{datarate=76Mbps;} <--> sdnnode[37].port++;
        sdnnode[33].port++ <--> Channel_c{datarate=79Mbps;} <--> sdnnode[44].port++;
        sdnnode[35].port++ <--> Channel_c{datarate=79Mbps;} <--> sdnnode[36].port++;
        sdnnode[36].port++ <--> Channel_c{datarate=72Mbps;} <--> sdnnode[39].port++;
        sdnnode[37].port++ <--> Channel_c{datarate=62Mbps;} <--> sdnnode[38].port++;
        sdnnode[38].port++ <--> Channel_c{datarate=67Mbps;} <--> sdnnode[39].port++;
        sdnnode[38].port++ <--> Channel_c{datarate=67Mbps;} <--> sdnnode[44].port++;
        sdnnode[39].port++ <--> Channel_c{datarate=72Mbps;} <--> sdnnode[40].port++;
        sdnnode[39].port++ <--> Channel_c{datarate=76Mbps;} <--> sdnnode[41].port++;
        sdnnode[39].port++ <--> Channel_c{datarate=63Mbps;} <--> sdnnode[42].port++;
        sdnnode[40].port++ <--> Channel_c{datarate=66Mbps;} <--> sdnnode[41].port++;
        sdnnode[41].port++ <--> Channel_c{datarate=61Mbps;} <--> sdnnode[42].port++;
        sdnnode[42].port++ <--> Channel_c{datarate=80Mbps;} <--> sdnnode[43].port++;
        sdnnode[42].port++ <--> Channel_c{datarate=69Mbps;} <--> sdnnode[44].port++;
        sdnnode[43].port++ <--> Channel_c{datarate=67Mbps;} <--> sdnnode[46].port++;
        sdnnode[43].port++ <--> Channel_c{datarate=63Mbps;} <--> sdnnode[47].port++;
        sdnnode[44].port++ <--> Channel_c{datarate=76Mbps;} <--> sdnnode[45].port++;
        sdnnode[45].port++ <--> Channel_c{datarate=79Mbps;} <--> sdnnode[46].port++;
        sdnnode[45].port++ <--> Channel_c{datarate=64Mbps;} <--> sdnnode[47].port++;
        sdnnode[45].port++ <--> Channel_c{datarate=69Mbps;} <--> sdnnode[48].port++;
        sdnnode[46].port++ <--> Channel_c{datarate=71Mbps;} <--> sdnnode[48].port++;
        sdnnode[47].port++ <--> Channel_c{datarate=61Mbps;} <--> sdnnode[48].port++;
        
        
        
        
        
        
        
        
      }
