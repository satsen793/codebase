                       
                                                                

class JointActionSpace:
    def __init__(self, jump_actions, content_actions):
                                                  
        self.action_pairs = [(j, c) for j in jump_actions for c in content_actions]
        self.index_to_action = {i: pair for i, pair in enumerate(self.action_pairs)}
        self.action_to_index = {pair: i for i, pair in enumerate(self.action_pairs)}

    def get_index(self, jump_action, content_action):
        return self.action_to_index.get((jump_action, content_action), None)

    def get_action(self, index):
        return self.index_to_action.get(index, None)

    def size(self):
        return len(self.action_pairs)

    def mask(self, jump_mask, content_mask):
                                                                            
                                                   
        mask = []
        for j, c in self.action_pairs:
            jm = jump_mask[j] if isinstance(j, str) else jump_mask.get(j, 0)
            cm = content_mask[c] if isinstance(c, str) else content_mask.get(c, 0)
            mask.append(jm and cm)
        return mask
