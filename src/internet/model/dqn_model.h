#ifndef DQN_MODEL_H_
#define DQN_MODEL_H_



#include "tensorflow/core/public/session.h"

#include <vector>
#include <string>

using namespace std;

class TrainingSet{

public:
	TrainingSet(vector<float> state, vector<float> newstate, uint32_t action, float reward, uint32_t done);

	vector<float> GetState();
	vector<float> GetNewState();

	uint32_t GetAction();
	float GetReward();
	uint32_t GetDone();



private:
	vector<float> state;
	vector<float> newstate;
	uint32_t action;
	float reward;
	uint32_t done;

};

class DQN_model{
public:
//	DQN_model();

	DQN_model(const string& graph_def_filename,
			uint32_t input_size, uint32_t output_size, float discount_factor);

	void Init();

	void Checkpoint(const tensorflow::string& checkpoint_prefix);

	void Restore(const string& checkpoint_prefix);

	uint32_t argmax(vector<float> v);

	vector<vector<float>> Predict(vector<vector<float>>& batch);

	void train(vector<TrainingSet*> t);

	void SetInputSize(int input_size);

	void SetOutputSize(int output_size);

	uint32_t GetInputSize();

	uint32_t GetOutputSize();

	uint32_t GetStepCount();

	void SetStepCount(uint32_t step_count);

	vector<float> GetPstate();

	vector<float> GetCstate();

	uint32_t GetAction();

	float GetReward();

	void SetPstate(vector<float> pState);

	void SetCstate(vector<float> cState);

	void SetAction(uint32_t action);

	void SetDone(uint32_t done);

	void SetReward(float reward);

	uint32_t GetDone();

	bool DirectoryExists(const string& dir) ;

private:
	tensorflow::Session *session;

	uint32_t input_size, output_size;
	uint32_t step_count = 0;

	float discount_factor;

	vector<float> pState, cState;
	uint32_t action = -1;
	float reward = 0;
	uint32_t done = 0;

	uint32_t epsilon_k  					{(10)};



private:
  	void SaveOrRestore(const string& checkpoint_prefix, const string& op_name);

	tensorflow::Tensor MakeTensor(const vector<vector<float>>& batch);

	tensorflow::Tensor MakeOutputTensor(const std::vector<vector<float>>& batch);


};

#endif
