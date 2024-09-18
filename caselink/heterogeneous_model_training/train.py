import torch
import torch.nn as nn
from tqdm import tqdm
import random
from torch_metrics import t_metrics, metric, yf_metric, rank
from CaseLink_model import RMSELoss, early_stopping
import os


def get_path():
    path = os.getcwd()
    path_clean = []
    for x in path.split('/'):
        if x != 'caselink_reproducibility':
            path_clean.append(x)
        else:
            path_clean.append(x)
            break
    path = '/'.join(path_clean)
    return path


def forward(dataset, model, device, writer, dataloader, data, label_dict, train_candidate_list, epoch, batch_size, temp, lamb, hard_neg_num, train_flag, test_mask, test_label_list, test_query_list, test_query_index_list, optimizer=None,  training_setup='tmp'):
    if train_flag:
        ## Training
        model.train()
        print('epoch: ', epoch)

        optimizer.zero_grad()
        loss_model = nn.CrossEntropyLoss()
        reg_loss_model = RMSELoss()
        loss_model.to(device)
        reg_loss_model.to(device)
        
        for query_index, labels, pos_case_index, ran_neg_index, hard_neg_index in tqdm(dataloader):
            x_dict = data.x_dict
            x_dict = {k: v.to(device) for k, v in x_dict.items()}
            edge_index_dict = data.edge_index_dict
            edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}
            train_graph_rep_matrix = model(x_dict, edge_index_dict)
            case_nums = data['case']['name_list']
            train_Case_rep_matrix = train_graph_rep_matrix['case']
            
            """
            train_graph_rep_matrix = model(data[0][0].to(device), data[0][0].ndata['feat'].to(device))
            case_nums = len(data[1]['case_name_list'])
            train_Case_rep_matrix = train_graph_rep_matrix[:case_nums,:]
            """

            ##REGloss
            train_Case_rep_matrix_norm = train_Case_rep_matrix / train_Case_rep_matrix.norm(dim=1)[:, None]
            new_graph_score_matrix = torch.mm(train_Case_rep_matrix_norm, train_Case_rep_matrix_norm.T)

            random_index1 = random.choices(train_candidate_list, k=batch_size)
            random_index2 = random.choices(train_candidate_list, k=batch_size)
            random_index1 = torch.Tensor(random_index1).int().to(device)
            random_index2 = torch.Tensor(random_index2).int().to(device)
            new_query_rep_matrix = new_graph_score_matrix[random_index1]
            new_query_rep_matrix = new_query_rep_matrix[:,random_index2]
            
            l_reg = new_query_rep_matrix.mean()

            ## NCEloss
            train_query_graph_matrix = train_Case_rep_matrix[query_index]
            train_query_graph_matrix_norm = train_query_graph_matrix / train_query_graph_matrix.norm(dim=1)[:, None]

            pos_graph_matrix = train_Case_rep_matrix[pos_case_index]
            pos_graph_matrix_norm = pos_graph_matrix / pos_graph_matrix.norm(dim=1)[:, None]

            ran_neg_graph_matrix = train_Case_rep_matrix[ran_neg_index]
            ran_neg_graph_matrix_norm = ran_neg_graph_matrix / ran_neg_graph_matrix.norm(dim=1)[:, None]

            hard_neg_index = [x for a in hard_neg_index for x in a]
            hard_neg_index = torch.Tensor(hard_neg_index).int()
            hard_neg_graph_matrix = train_Case_rep_matrix[hard_neg_index]
            hard_neg_graph_norm = hard_neg_graph_matrix / hard_neg_graph_matrix.norm(dim=1)[:, None]        

            l_que_neg = torch.mm(train_query_graph_matrix_norm, train_query_graph_matrix_norm.transpose(0,1))

            l_pos = torch.mm(train_query_graph_matrix_norm, pos_graph_matrix_norm.transpose(0,1))

            l_ran_neg = torch.mm(train_query_graph_matrix_norm, ran_neg_graph_matrix_norm.transpose(0,1))

            l_hard_neg = torch.mm(train_query_graph_matrix_norm, hard_neg_graph_norm.transpose(0,1))

            if hard_neg_num != 0:
                logits = torch.cat([l_pos, l_que_neg, l_ran_neg, l_hard_neg], dim=1).to(device)
            else:
                logits = torch.cat([l_pos, l_que_neg, l_ran_neg], dim=1).to(device)

            logits_label = torch.arange(0, len(labels)).type(torch.LongTensor).to(device)
            l_nce = loss_model(logits/temp, logits_label)
            
            # loss = l_nce
            
            loss = lamb*l_reg + l_nce
            
            writer.add_scalar('Train/NCELoss', l_nce.item(), epoch)
            # print('Train/NCELoss:', l_nce.item())

            writer.add_scalar('Train/REGLoss', l_reg.item(), epoch)
            # print('Train/REGLoss:', l_reg.item())

            writer.add_scalar('Train/Loss', loss.item(), epoch)
            print('Train/Loss:', loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
                
    else:
        ## Test
        model.eval()
        with torch.no_grad():
            test_x_dict = data.x_dict
            test_x_dict = {k: v.to(device) for k, v in test_x_dict.items()}
            test_edge_index_dict = data.edge_index_dict
            test_edge_index_dict = {k: v.to(device) for k, v in test_edge_index_dict.items()}
            test_graph_rep_matrix = model(test_x_dict, test_edge_index_dict)
            case_nums = data['case']['name_list']
            test_Case_rep_matrix = test_graph_rep_matrix['case']
            
            """
            test_graph_rep_matrix = model(data[0][0].to(device), data[0][0].ndata['feat'].to(device))
            case_nums = len(data[1]['case_name_list'])
            test_Case_rep_matrix = test_graph_rep_matrix[:case_nums,:]
            """

            test_Case_rep_matrix_norm = test_Case_rep_matrix / test_Case_rep_matrix.norm(dim=1)[:, None]

            test_score_matrix = torch.mm(test_Case_rep_matrix_norm, test_Case_rep_matrix_norm.T)
            test_score_matrix.fill_diagonal_(float('-inf'))
            
            sim_score = []
            for i in test_query_index_list:
                query_index = i
                score = test_score_matrix[query_index, :]
                sim_score.append(score)
            test_sim_score = torch.stack(sim_score)

            nominator = torch.log((torch.exp(test_sim_score / temp) * ((test_mask == 1) + 1e-24)).sum(dim=1))
            denominator = torch.logsumexp(test_sim_score / temp, dim=1)
            loss = -(nominator - denominator)
            loss = loss.mean()
            
            
            print("Test/Loss:", loss)
            writer.add_scalar('Test/Loss', loss.item(), epoch)

            final_pre_dict = rank(test_sim_score, len(test_label_list), test_query_list, test_label_list)

            correct_pred, retri_cases, relevant_cases, Micro_pre, Micro_recall, Micro_F, macro_pre, macro_recall, macro_F = metric(5, final_pre_dict, label_dict)
            # yf_dict, correct_pred_yf, retri_cases_yf, relevant_cases_yf, Micro_pre_yf, Micro_recall_yf, Micro_F_yf, macro_pre_yf, macro_recall_yf, macro_F_yf = yf_metric(5, yf_path, final_pre_dict, label_dict)

            ndcg_score, mrr_score, map_score, p_score = t_metrics(label_dict, final_pre_dict, 5)
            # ndcg_score_yf, mrr_score_yf, map_score_yf, p_score_yf = t_metrics(label_dict, yf_dict, 5)

            print("Test result:")
            print("Correct Predictions: ", correct_pred)
            print("Retrieved Cases: ", retri_cases)
            print("Relevant Cases: ", relevant_cases)

            print("Micro Precision: ", Micro_pre)
            print("Micro Recall: ", Micro_recall)
            print("Micro F1: ", Micro_F)

            print("Macro Precision: ", macro_pre)
            print("Macro Recall: ", macro_recall)
            print("Macro F1: ", macro_F)

            print("NDCG@5: ", ndcg_score)
            print("MRR@5: ", mrr_score)
            print("MAP: ", map_score)

            '''
            print("Correct Predictions yf: ", correct_pred_yf)
            print("Retrived Cases yf: ", retri_cases_yf)
            print("Relevant Cases yf: ", relevant_cases_yf)

            print("Micro Precision yf: ", Micro_pre_yf)
            print("Micro Recall yf: ", Micro_recall_yf)
            print("Micro F1 yf: ", Micro_F_yf)

            print("Macro Precision yf: ", macro_pre_yf)
            print("Macro Recall yf: ", macro_recall_yf)
            print("Macro F1 yf: ", macro_F_yf)

            print("NDCG@5 yf: ", ndcg_score_yf)
            print("MRR@5 yf: ", mrr_score_yf)
            print("MAP yf: ", map_score_yf)
            '''

            writer.add_scalar("One stage/Correct num", correct_pred, epoch)        
            writer.add_scalar("One stage/Micro Precision", Micro_pre, epoch)
            writer.add_scalar("One stage/Micro Recall", Micro_recall, epoch)
            writer.add_scalar("One stage/Micro F1", Micro_F, epoch)

            writer.add_scalar("One stage/Macro Precision", macro_pre, epoch)
            writer.add_scalar("One stage/Macro Recall", macro_recall, epoch)
            writer.add_scalar("One stage/Macro F1", macro_F, epoch)


            writer.add_scalar("One stage/NDCG@5", ndcg_score, epoch)
            writer.add_scalar("One stage/MRR", mrr_score, epoch)
            writer.add_scalar("One stage/MAP", map_score, epoch)

            '''
            writer.add_scalar("One stage yf/Correct num yf", correct_pred_yf, epoch)        
            writer.add_scalar("One stage yf/Micro Precision yf", Micro_pre_yf, epoch)
            writer.add_scalar("One stage yf/Micro Recall yf", Micro_recall_yf, epoch)
            writer.add_scalar("One stage yf/Micro F1 yf", Micro_F_yf, epoch)

            writer.add_scalar("One stage yf/Macro Precision yf", macro_pre_yf, epoch)
            writer.add_scalar("One stage yf/Macro Recall yf", macro_recall_yf, epoch)
            writer.add_scalar("One stage yf/Macro F1 yf", macro_F_yf, epoch)


            writer.add_scalar("One stage yf/NDCG@5 yf", ndcg_score_yf, epoch)
            writer.add_scalar("One stage yf/MRR yf", mrr_score_yf, epoch)
            writer.add_scalar("One stage yf/MAP yf", map_score_yf, epoch)
            '''
            
            predict_path = get_path() + '/datasets/' + dataset + '/caselink_heterogeneous_experiments/'
            with open(predict_path + training_setup + '.txt', "a") as fOut:
                fOut.write(10*'*' + f' {epoch} ' + 10*'*' + '\n')
                fOut.write('\n')
                
                fOut.write("Correct Predictions: "+str(correct_pred)+'\n')
                fOut.write("Retrived Cases: "+str(retri_cases)+'\n')
                fOut.write("Relevant Cases: "+str(relevant_cases)+'\n')

                fOut.write("Micro Precision: "+str(Micro_pre)+'\n')
                fOut.write("Micro Recall: "+str(Micro_recall)+'\n')
                fOut.write("Micro F1: "+str(Micro_F)+'\n')

                fOut.write("Macro Precision: "+str(macro_pre)+'\n')
                fOut.write("Macro Recall: "+str(macro_recall)+'\n')
                fOut.write("Macro F1: "+str(macro_F)+'\n')

                fOut.write("NDCG@5: "+str(ndcg_score)+'\n')
                fOut.write("MRR@5: "+str(mrr_score)+'\n')
                fOut.write("MAP: "+str(map_score)+'\n')

                '''
                fOut.write("Correct Predictions yf: "+str(correct_pred_yf)+'\n')
                fOut.write("Retrived Cases yf: "+str(retri_cases_yf)+'\n')
                fOut.write("Relevant Cases yf: "+str(relevant_cases_yf)+'\n')

                fOut.write("Micro Precision yf: "+str(Micro_pre_yf)+'\n')
                fOut.write("Micro Recall yf: "+str(Micro_recall_yf)+'\n')
                fOut.write("Micro F1 yf: "+str(Micro_F_yf)+'\n')

                fOut.write("Macro Precision yf: "+str(macro_pre_yf)+'\n')
                fOut.write("Macro Recall yf: "+str(macro_recall_yf)+'\n')
                fOut.write("Macro F1 yf: "+str(macro_F_yf)+'\n')

                fOut.write("NDCG@5 yf: "+str(ndcg_score_yf)+'\n')
                fOut.write("MRR@5 yf: "+str(mrr_score_yf)+'\n')
                fOut.write("MAP yf: "+str(map_score_yf)+'\n')
                '''

                fOut.write('\n')
                
                fOut.close()
        
        return ndcg_score
        