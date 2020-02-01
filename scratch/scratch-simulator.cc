/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#include <iostream>

#include "ns3/core-module.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

using namespace ns3;
using namespace std;
using namespace tensorflow;

NS_LOG_COMPONENT_DEFINE ("ScratchSimulator");

int 
main (int argc, char *argv[])
{
  NS_LOG_UNCOND ("Scratch Simulator");

Session *session; 
	Status status = NewSession(SessionOptions(), &session);
	if(!status.ok()){

		cout << "1 : " << status.ToString() << "\n";
		return 1;
	}

	// Read protobuf graph 
	GraphDef graph_def; 
	status = ReadBinaryProto(Env::Default(), "tensorflow_model/graph.pb", &graph_def);
	if(!status.ok()){

		cout << "2 : " << status.ToString() << "\n";
		return 1;
	}

	status = session -> Create(graph_def);
	if(!status.ok()){

		cout << "3 : " << status.ToString() << "\n";
		return 1;
	}


  Simulator::Run ();
  Simulator::Destroy ();
}
