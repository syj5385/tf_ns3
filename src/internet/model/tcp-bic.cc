/*
 * Implemented by YeongJun
 * 1st Feb 2020
 */
#include <iostream>
#include <ctime>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>


#include "tcp-bic.h"
#include "dqn_model.h"
#include "ns3/log.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/random-variable-stream.h"
#include "ns3/nstime.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

const string graph_def_filename = "./tensorflow_model/graph.pb";
const string checkpoint_dir ="./tensorflow_model/checkpoint";
const string checkpoint_prefix = checkpoint_dir + "/checkpoint";

const bool train_trace = true;

const uint32_t input_size = 2;
const uint32_t output_size = 3;
const uint32_t train_chunk_size = 10;
const float discount_factor = 0.9;

const uint32_t max_episode = 100;
const uint32_t max_step_in_one_episode = 10;
const uint32_t trainTime_ms = 100;
const uint32_t train_chunk = 1;
const float perf_limit = 0;
const uint32_t alpha = 1;
const uint32_t beta = 2;
const uint32_t epsilon = 10;

const uint32_t maxCwnd = 1000.0;
const float maxRtt = 500000.0;


const string throu_file = "throughput.xls";
ofstream writeFile(throu_file.data());

enum {
	CWND_UP = 0,
	CWND_DOWN,
	CWND_NOTHING
};


/* TrainingSet implementation*/
TrainingSet::TrainingSet(vector<float> state, vector<float> newstate, uint32_t action, float reward, uint32_t done){
	this -> state = state;
	this -> newstate = state;
	this -> action = action;
	this -> reward = reward;
	this -> done = done;
}

vector<float> TrainingSet::GetState(){
	return state;
}

vector<float> TrainingSet::GetNewState(){
	return newstate;
}

uint32_t TrainingSet::GetAction(){
	return action;
}

float TrainingSet::GetReward(){
	return reward;
}

uint32_t TrainingSet::GetDone(){
	return done;
}

DQN_model::DQN_model(const string& graph_def_filename, uint32_t input_size, uint32_t output_size, float discount_factor)
{
	NS_LOG_UNCOND ("Initialize DQN model [" << input_size << " ==> [Hidden Layer] ==> " << output_size << "]\n");

	this -> input_size = input_size;
	this -> output_size = output_size;
	this -> discount_factor = discount_factor;
	this -> step_count = 0;
	this -> action = -1;

	tensorflow::GraphDef graph_def;
	TF_CHECK_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
	                                            graph_def_filename, &graph_def));
	TF_CHECK_OK(tensorflow::NewSession(tensorflow::SessionOptions(), &session));
	TF_CHECK_OK(session->Create(graph_def));

	bool restore = DirectoryExists(checkpoint_dir);
	if (!restore) {
		std::cout << "Initializing model weights\n";
	    Init();
	} else {
	    std::cout << "Restoring model weights from checkpoint\n";
	    Restore(checkpoint_prefix);
	}
}

void DQN_model::Init(){
	TF_CHECK_OK(session->Run({}, {}, {"init"}, nullptr));

}

void DQN_model::Checkpoint(const tensorflow::string& checkpoint_prefix){
	SaveOrRestore(checkpoint_prefix, "save/control_dependency");
}

void DQN_model::SaveOrRestore(const string& checkpoint_prefix, const string& op_name) {
    tensorflow::Tensor t(tensorflow::DT_STRING, tensorflow::TensorShape());
    t.scalar<string>()() = checkpoint_prefix;
    TF_CHECK_OK(session->Run({{"save/Const", t}}, {}, {op_name}, nullptr));

}
void DQN_model::Restore(const string& checkpoint_prefix){
	SaveOrRestore(checkpoint_prefix, "save/restore_all");
}

uint32_t DQN_model::argmax(vector<float> v){
	auto max = *max_element(v.begin(), v.end());
	auto it= find(v.begin(), v.end(), max);
	auto index = distance(v.begin(), it);

	return index;
}

tensorflow::Tensor DQN_model::MakeTensor(const vector<vector<float>>& batch){
	tensorflow::Tensor t(tensorflow::DT_FLOAT, tensorflow::TensorShape({(int)batch.size(), GetInputSize()}));
	for(uint32_t i=0; i<batch.size(); i++){
		for(uint32_t j=0; j<GetInputSize(); j++){
			t.flat<float>()(i*GetInputSize() + j) = batch[i][j];
		}
	}
	return t;
}

tensorflow::Tensor DQN_model::MakeOutputTensor(const std::vector<vector<float>>& batch){
	tensorflow::Tensor t(tensorflow::DT_FLOAT, tensorflow::TensorShape({(int)batch.size(), GetOutputSize()}));
	for(uint32_t i=0; i<batch.size(); i++){
			for(uint32_t j=0; j<GetOutputSize(); j++){
				t.flat<float>()(i*GetOutputSize() + j) = batch[i][j];
			}
		}
	return t;
}

vector<vector<float>> DQN_model::Predict(vector<vector<float>>& batch){

	/* predict input and output always has one row */

	vector<vector<float>> out;
	vector<tensorflow::Tensor> out_tensors;

	TF_CHECK_OK(session->Run({{"input", MakeTensor(batch)}}, {"output"}, {}, &out_tensors));

	uint32_t h = out_tensors[0].flat<float>().size() / GetOutputSize();
	out.resize(h);

	for(uint32_t i=0; i<h; i++){
		for(uint32_t j=0; j<GetOutputSize() ;j++){
			out[i].push_back(out_tensors[0].flat<float>()(GetOutputSize()*i+j));
		}
	}

	return out;
}

void DQN_model::train(vector<TrainingSet*> t){
	vector<vector<float>> Qinput, Qinput_new;
	vector<vector<float>> Qpred, Qpred_new;

	for(uint32_t i=0; i<t.size(); i++){ // row #
		Qinput.push_back(t[i] -> GetState());

		Qinput_new.push_back(t[i] -> GetNewState());
	}

	Qpred = Predict(Qinput);
	Qpred_new = Predict(Qinput_new);

	for(uint32_t i=0; i<t.size(); i++){
		if(t[i] -> GetDone() == 1){
			Qpred[i].at(GetAction()) = t[i] -> GetReward();
		}
		else if(t[i] ->GetDone() == 0){
			Qpred[i].at(GetAction()) = t[i] -> GetReward() +
					discount_factor * (*max_element(Qpred_new[i].begin(), Qpred_new[i].end()));
		}
	}

	TF_CHECK_OK(session->Run({{"input", MakeTensor(Qinput)}, {"target", MakeOutputTensor(Qpred)}},
					  {}, {"train"}, nullptr));
}

void DQN_model::SetInputSize(int input_size){}

void DQN_model::SetOutputSize(int output_size){}

uint32_t DQN_model::GetInputSize(){ return input_size; }

uint32_t DQN_model::GetOutputSize(){ return output_size; }

void DQN_model::SetStepCount(uint32_t step_count){ this->step_count = step_count; }

uint32_t DQN_model::GetStepCount(){ return this->step_count; }

vector<float> DQN_model::GetPstate(){ return pState; }

vector<float> DQN_model::GetCstate(){ return cState; }

uint32_t DQN_model::GetAction(){ return action; }

uint32_t DQN_model::GetDone(){ return action; }

float DQN_model::GetReward(){ return reward; }

void DQN_model::SetPstate(vector<float> pState){ this -> pState = pState; }

void DQN_model::SetCstate(vector<float> cState){ this -> cState = cState; }

void DQN_model::SetAction(uint32_t action){ this -> action = action; }

void DQN_model::SetReward(float reward){ this -> reward = reward; }

void DQN_model::SetDone(uint32_t done){ this -> done = done; }

bool DQN_model::DirectoryExists(const string& dir) {
  struct stat buf;
  return stat(dir.c_str(), &buf) == 0;
}


/* DRL Congestion control */
namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TcpBic");
NS_OBJECT_ENSURE_REGISTERED (TcpBic);

TypeId
TcpBic::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpBic")
    .SetParent<TcpCongestionOps> ()
    .AddConstructor<TcpBic> ()
    .SetGroupName ("Internet")
  ;
  return tid;
}


TcpBic::TcpBic () : TcpCongestionOps (),
		model(graph_def_filename, input_size, output_size, discount_factor)
{
	NS_LOG_FUNCTION (this);
//	model = new DQN_mode:l(graph_def_filename, input_size, output_size, discount_factor);

	srand((unsigned int)Simulator::Now().GetNanoSeconds());

}

TcpBic::TcpBic (const TcpBic &sock)  : TcpCongestionOps (sock),
		model(graph_def_filename, input_size, output_size, discount_factor)
{
	NS_LOG_FUNCTION (this);

	Uprev = 0;
	srand((unsigned int)Simulator::Now().GetNanoSeconds());
}


void
TcpBic::IncreaseWindow (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
  NS_LOG_FUNCTION (this << tcb << segmentsAcked);

}

uint32_t
TcpBic::Update (Ptr<TcpSocketState> tcb)
{
  NS_LOG_FUNCTION (this << tcb);

//  uint32_t segCwnd = tcb->GetCwndInSegments ();
  uint32_t cnt = 0;


  //NS_LOG_UNCOND("cwnd : " << segCwnd);

  return cnt;
}

vector<float> makeState(Ptr<TcpSocketState> tcb, const Time& rtt){
	vector<float> v;
	cout << "Current Cwnd : " << tcb -> GetCwndInSegments() << "\n";
//	v.push_back(tanh((float)tcb->GetCwndInSegments() / 300));
	v.push_back((float)tcb->GetCwndInSegments() / maxCwnd);
	v.push_back((float)rtt.GetMicroSeconds()*1000/maxRtt);
	return v;
}

void TcpBic::CCA_train(Ptr<TcpSocketState> tcb, const Time& rtt, DQN_model model){
	NS_LOG_FUNCTION (this << tcb);

}

void TcpBic::PktsAcked (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt)
{
	Time now = Simulator::Now();

	if(now.GetMilliSeconds() - lastTrainTime.GetMilliSeconds() > trainTime_ms ){

		uint32_t iter = step / max_step_in_one_episode;
		float e = 1.0 / ((iter/epsilon) + 1);
		float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

		if((int)model.GetAction () != -1){
			model.SetPstate(model.GetCstate()); // update state
			model.SetCstate(makeState(tcb, rtt));

			TrainingSet *t = new TrainingSet (model.GetPstate(), model.GetCstate(),
					model.GetAction(), model.GetReward(), model.GetDone());

			t_set.push_back(t);

			/* Calculate reward */
			float est_throughput = (((maxTxtmp - lastTx) *  8 / 1000.0) * 1000 / trainTime_ms);

			float U = (est_throughput / 100) - (rtt.GetMicroSeconds() / maxRtt);
			cout << "throughput : " << est_throughput << "\trtt : "
					<< (float)rtt.GetMicroSeconds()/1000.0 << "\n";

			cout << "PrevU : " << Uprev << "\tUtility = " << U << "[" << (U-Uprev) << "]\n";

			if(U - Uprev > perf_limit){
				model.SetReward(2);
			}
			else if(U - Uprev < (-1)*perf_limit){
				model.SetReward(-1);
			}
			else{
				model.SetReward(0);
			}
			/* Training */
			if(train_trace){
				cout << "[Training...]" << "iter : " << iter << "\tstep : << " << (step % max_step_in_one_episode) << "++++++++++++++++++++++++++++++++++++++++++++++++\n";
				cout << "Action : " << model.GetAction();
				cout << "\tReward : " << model.GetReward() << "\n\n";
				cout << "maxTx : " << maxTxtmp << "\t lastTx : " << lastTx << "\n";
				cout << "Sent : " << (((maxTxtmp - lastTx) *  8 / 1000) * 1000 / trainTime_ms) << "Kbps\n";
				if(writeFile.is_open()){
					writeFile << (((maxTxtmp - lastTx) *  8 / 1000) * 1000 / trainTime_ms) << "\n";
				}

				vector<vector<float>> input, input_new;
				input.push_back(model.GetCstate());
				input_new.push_back(model.GetPstate());

				vector<vector<float>> output, output_new;
				output = model.Predict(input);
				output_new = model.Predict(input_new);

				cout << "State " << model.GetPstate()[0] << "\t: [";
				for(uint32_t i=0; i<model.GetOutputSize(); i++){
					cout << output[0][i] << ",";
				}
				cout << "]\n";

				cout << "State " << model.GetCstate()[0] << "\t: [";
				for(uint32_t i=0; i<model.GetOutputSize(); i++){
					cout << output_new[0][i] << ",";
				}
				cout << "]\n";
			}

			/* Training will be executed in every [train_chunk] */
			if(t_set.size() == train_chunk){
				model.train(t_set);

				for(uint32_t i=0; i<t_set.size(); i++)
					delete t_set[i];

				t_set.clear();
				t_set.resize(0);

			}
			lastTx = maxTxtmp ;
			Uprev = U;
		}
		else{
			model.SetCstate(makeState(tcb, rtt));
			lastTx = tcb -> m_highTxMark;

			/* Calculate reward */

			Uprev = 0;
		}

		/* Select Action */
		if( r < e){
			// Random sampleing
			cout << "\t==>> Random\n";
			model.SetAction((uint32_t)rand() % 3);
		}
		else{
			// GetAction from Q
			cout << "\t==>> Selection\n";
			vector<vector<float>> input;
			input.push_back(model.GetCstate());
			model.SetAction(model.argmax(model.Predict(input)[0]));
		}

		cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n";
		switch(model.GetAction()){
		case CWND_UP :
			tcb -> m_cWnd += (3*tcb -> m_segmentSize);
			break;

		case CWND_DOWN :
			tcb -> m_cWnd -= (2*tcb -> m_segmentSize);
			break;

		}


		/* Update step counter */
		step++;
		if(step > (max_step_in_one_episode * max_episode)){
			/* End of 1 training */
			step = 0;
			cout << "Store Checkpoint\n";
			model.Checkpoint(checkpoint_prefix);
		}

		lastTrainTime = now;
	}

	if(tcb -> m_highTxMark > maxTxtmp){
		maxTxtmp = tcb -> m_highTxMark;
	}
}

std::string
TcpBic::GetName () const
{
  return "TcpBic";
}

uint32_t
TcpBic::GetSsThresh (Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight)
{
	NS_LOG_FUNCTION (this << tcb << bytesInFlight);
	return std::max (2 * tcb->m_segmentSize, bytesInFlight / 2);
}

Ptr<TcpCongestionOps>
TcpBic::Fork (void)
{
  return CopyObject<TcpBic> (this);
}

} // namespace ns3
