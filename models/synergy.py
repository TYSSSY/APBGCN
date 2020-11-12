from models.pairs import Pairs
import torch
def get_synergy(data):
    get_pair = Pairs()
    frames = data.size(0)
    synergy_matrix = torch.zeros(0, frames).cuda()
    #synergy_matrix_2 = torch.zeros(0, frames)
    for joint_1, joint_2 in get_pair.total_collection:
        # pair : (joint 1, joint 2)
        pair_synergy = torch.mul(data[:,joint_1,7], data[:,joint_2,7]) + torch.mul(data[:,joint_1,8],
                data[:,joint_2,8]) + torch.mul(data[:,joint_1,9], data[:,joint_2,9])
        #print(pair_synergy.shape)
        #pair_synergy_person_1 = torch.unsqueeze(pair_synergy.permute(1, 0)[0], 0)
        
        #pair_synergy_person_2 = torch.unsqueeze(pair_synergy.permute(1, 0)[1], 0)
        #synergy_matrix_1 = torch.cat((synergy_matrix_1, pair_synergy_person_1))
        #synergy_matrix_2 = torch.cat((synergy_matrix_2, pair_synergy_person_2))

        synergy_matrix = torch.cat((synergy_matrix, torch.unsqueeze(pair_synergy,0)),dim=0)
    
    return synergy_matrix
