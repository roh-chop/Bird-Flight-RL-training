import rlenv
import numpy as np
import tensorflow as tf
from tensorflow import math


    
###hyperparams

updates = 50
timesteps = 2048

optimizera = tf.keras.optimizers.Adam(learning_rate = 3e-5)


Critic = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128,activation='tanh',input_shape = (18,)),
    tf.keras.layers.Dense(128,activation='tanh'),
    tf.keras.layers.Dense(1)
])

#Critic.load_weights('weight1.weights.h5')

loss_fn = tf.keras.losses.MeanSquaredError()
Critic.compile(optimizer=optimizera, loss = loss_fn)


def log_normpdf(x, mean, log_sd):  
    x = tf.cast(x,tf.float64)
    mean = tf.cast(mean,tf.float64)
    log_sd = tf.cast(log_sd,tf.float64)
    prob = -0.5 * (((x-mean)/(math.exp(log_sd)+1e-8))**2 + 2*log_sd + tf.cast(math.log(2*np.pi),tf.float64))
    return prob

total_reward = 0

state = rlenv.init()
    
class PPO_Clip(tf.keras.losses.Loss):
    def call(self,y_pred,y_true):
        new_probs,stdevs = y_pred
        old_probs,advantages = y_true
        
        entropies = 0.5*math.log(np.pi*np.e*tf.pow(math.exp(stdevs),2)+1e-8)
        entropy = 0*tf.reduce_mean(entropies)
        ratio = tf.reduce_prod(math.exp(new_probs-old_probs),axis=1)
        loss = math.minimum(ratio*advantages,tf.clip_by_value(ratio,0.8,1.2)*advantages)
        loss = -tf.reduce_mean(loss)-entropy
        return loss
    
optimizer = tf.keras.optimizers.Adam(learning_rate = 3e-5) 
objective = PPO_Clip()

        
class ActorContinuous():
    def __init__(self,in_shape,n_actions):
        self.model = tf.keras.models.Sequential([
                                        tf.keras.layers.Dense(128,activation='tanh',input_shape = (in_shape,)),
                                        tf.keras.layers.Dense(128,activation='tanh'),
                                        tf.keras.layers.Dense(n_actions*2)
                                        ])
        #self.model.load_weights("weight.weights.h5")
        self.input_shape = in_shape
        self.n_actions = n_actions
        
    def forward_internal(self,state):
        action_dist = self.model(state)
        action_stds = tf.clip_by_value(action_dist[:,1::2],-0.5,0.5)
        action_means = tf.tanh(action_dist[:,0::2])
        action_out = tf.stack([action_means,action_stds],axis=2)
        return tf.reshape(action_out,(-1,2*self.n_actions))
                
    def forward(self,state):
        action_dist = self.forward_internal(state)
        act_vals = np.zeros(self.n_actions)
        log_probs = np.zeros(self.n_actions)
        
        for i in range(self.n_actions):
            act_vals[i] = tf.random.normal((),action_dist[0][i*2],math.exp(action_dist[0][i*2+1]))
            log_probs[i]= log_normpdf(act_vals[i],action_dist[0][i*2],action_dist[0][i*2+1])
        return(act_vals,log_probs)
    
    def train(self,states,action_vals,log_old_probs,advantages):
        for k in range(10):
            shuffle = tf.cast(tf.random.shuffle(tf.range(timesteps)),tf.int32).numpy().tolist()
            for l in range(timesteps//64):
                batch_inds = shuffle[l*64:(l+1)*64]
                batch = tf.gather(states,batch_inds)
                with tf.GradientTape() as tape:
                    action_dist = self.forward_internal(batch)
                    new_means = action_dist[:,0::2]
                    new_stds = action_dist[:,1::2]
                    log_new_probs = log_normpdf(tf.gather(action_vals,batch_inds),new_means,new_stds)
                    loss = objective((log_new_probs,new_stds),(tf.gather(log_old_probs,batch_inds),tf.gather(advantages,batch_inds)))
                    
                grad = tape.gradient(loss,self.model.trainable_variables)
                optimizer.apply_gradients(zip(grad,self.model.trainable_variables))
                if(l==(32-1)):
                    print(f"loss:{loss}")
                    
    def save_weights(self,file):
        self.model.save_weights(file)
    
Actor = ActorContinuous(18,5)
    
for i in range(updates):
  reset = True
  gamma = 0.99
  step = 0
  
  batch = []
  values = []
  
  ###collection

  for j in range(timesteps):
    if(reset):
      print(f"reward: {total_reward}")
      step = 0
      gamma = 0.99
      total_reward=0
      reset = False
      
    
    (act_vals,log_probs) = Actor.forward(tf.convert_to_tensor([state]))
    

    
    
   
    state_save = state
    
    value = Critic(tf.convert_to_tensor([state]))
    values.append(value[0].numpy()[0])
    
    out = rlenv.sim(act_vals)
    
    state = out.state
    reward = out.reward
    terminated = out.terminated
    truncated = out.truncated
    step = step+1
    

    
    
    if terminated or truncated:
        reset = True

    
    total_reward+=reward
    
    
    data = [reward, *state_save, *act_vals, *log_probs]
    batch.append(data)
  
    
    
    for k in range(step-1):
        batch[j-k-1][0]+=reward*(gamma**(k+1))
    

  total_reward=0
  batch = np.array(batch)
     
  rewards = tf.cast(tf.convert_to_tensor(batch[:,0]),tf.float64)
  states = tf.convert_to_tensor(batch[:,1:len(state_save)+1])  
  action_vals = tf.convert_to_tensor(batch[:,len(state_save)+1:len(state_save)+len(act_vals)+1])
  log_old_probs = tf.convert_to_tensor(batch[:,len(state_save)+len(act_vals)+1:len(batch[0])])
  values = tf.cast(tf.convert_to_tensor(values),tf.float64)
  advantages_i = rewards-values
  advantages = (advantages_i-tf.reduce_mean(advantages_i))/math.reduce_std(advantages_i)
  Critic.fit(states,rewards,epochs=10,batch_size = 64)
  Actor.train(states,action_vals,log_old_probs,advantages)
  
Actor.save_weights('weight.weights.h5')
Critic.save_weights('weight1.weights.h5')


    
        
    
        
