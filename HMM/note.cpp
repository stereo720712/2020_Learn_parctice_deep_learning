//http://www.csie.ntnu.edu.tw/~u91029/HiddenMarkovModel.html

//Transition Probability, A ,
//is probability matrix  [N,N]




//Initial Probability, Π



//=================================Markov Model

const int N = 3; //path total state number 
double a[N][N]; // all node transition probability matrix => Markov Model
double pai[N]; // initial probability list Π

// for a full path probability , q:state sequence, T: total walked states 
double probability_MM(int* q, int T){
	// q is states node list(sequence) ,total states num is T(PATH is T -1) ,
	double p = pai[q[0]]; // start walk first state probability

	for(int i = 1; i < T; ++i){	
		//the probability for the path which walked through states in T
		p *= a[q[i-1]][q[i]];
		return q;
	}

}


//================== Hidden Markdov Model
//Key:
//Every state throws one or more value for user when user visit a state
//The vaule we define every state can throws M kind value,(set by user or defined ,ex: when raining, mary go to A place is 0.3,stay at home is 0.6)
//Every state throws some value have it's own probability
//State ===> throws Value's probability mark bi(k)
//every state has a set B for Vi(as N functions , total is NxM Matrix)

const int N = 3, M = 3, T = 15 //N:number of states , M:every state can throw M value, T:Total walks states
double pi[N] //all state initial probability
double a[N][N] //all paths probability for each states
double b[N][M]// all states value throw probability store matrix
//HMM  model means above declare
// q:state sequence, o:ovservation sequence(the list store every step the value which displayed by the state),T:
double probability_HMM(int* q,int* o , int T){

	//first step probability
	double p = pi[q[0]] * b[q[0]][o[0]]; //state happen probability * the value throws probability
	for(int i=1;i<T,++i){
		p *= a[q[i-1]][q[i]] * b[q[i]][o[i]]
	}
	return p;

}
