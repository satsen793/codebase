            
                                                  

class Trainer:
    def __init__(self, env, agent, teacher, safety_monitor):
        self.env = env
        self.agent = agent
        self.teacher = teacher
        self.safety_monitor = safety_monitor

    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                              
                state_vec = self.agent.encoder.encode(*state)
                                                 
                jump_action = self.agent.jump_policy.select_action(state_vec)
                content_action = self.agent.content_policy.select_action(state_vec)
                                  
                next_state, reward, done, info = self.env.step(jump_action, content_action)
                                        
                shaped_reward = self.teacher.total_reward(*reward)
                                   
                if info.get('mistake'):
                    count = self.safety_monitor.record_mistake(info['student_id'])
                    if self.safety_monitor.should_flag(info['student_id']):
                        print(f"Flag: Student {info['student_id']} reached mistake limit!")
                        self.safety_monitor.reset(info['student_id'])
                state = next_state
