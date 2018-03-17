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

#ifndef __NETWORKSIMULATOR_NROUTING_H_
#define __NETWORKSIMULATOR_NROUTING_H_

#include <omnetpp.h>
#include "DataPacket_m.h"
#include "Statistic.h"

using namespace std;
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstring>
#include <string>


/**
 * TODO - Generated class
 */
class Routing : public cSimpleModule
{
  private:
    int numPorts;
    int id;
    int numTx;
    int numNodes;
    vector<vector<int> >outPort;
    string folderName;
    int flow_num;



  public:
    Routing();
    virtual ~Routing();

  protected:
    virtual void initialize();
    virtual void handleMessage(cMessage *msg);
    void getRoutingInfo(int id, int flow_id, int rData[]);

};

#endif
