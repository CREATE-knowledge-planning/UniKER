from pathlib import Path
import sys
import os
import shutil
import numpy as np

from uniker.fc import fc
import uniker.fc.potential_useful_triples as pt


def check_path(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def run_kge(output_path: Path, data_path: Path, train_file_path: Path, model='TransE', model_name='kge_model', 
            kge_batch=512, kge_neg=128, kge_dim=500, kge_gamma=6.0, kge_alpha=0.5, kge_lr = 0.0005, 
            kge_iters=10000, kge_tbatch=16, use_cuda=True, cuda_visible_devices="2"):
    if use_cuda:
        #os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        cmd = 'python {} --cuda --do_train --do_valid --do_test ' \
              '--model {} --data_path {} -b {} -n {} -d {} -g {} ' \
              '-a {} -adv -lr {} --max_steps {} --test_batch_size {} ' \
              '-save {} --train_path {}'.format(
            Path("../UniKER/uniker/kge/run.py").resolve(), model, data_path.resolve(), 
            kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch,
            output_path.resolve() / model_name, train_file_path.resolve())
    else:
        cmd = 'python {} --do_train --do_valid --do_test ' \
              '--model {} --data_path {} -b {} -n {} -d {} -g {} ' \
              '-a {} -adv -lr {} --max_steps {} --test_batch_size {} ' \
              '-save {} --train_path {}'.format(
            Path("./../UniKER/uniker/kge/run.py").resolve(), model, data_path.resolve(), 
            kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch,
            output_path.resolve() / model_name, train_file_path.resolve())
    os.system(cmd)


def run_fc(data_folder, train_file_path, save_file_path, mln_rules):
    FC = fc.ForwardChain(data_folder, train_file_path, save_file_path, mln_rules)
    FC.run()


def eval_and_eliminate(output_path: Path, data_path: Path, k, model, model_name, train_name, save_name, 
                       noise_threshold=0.1, use_cuda=True, cuda_visible_devices="2"):
    # read train.txt, write new_train.txt
    # read infer.txt, write new_infer.txt

    # print('eval_and_eliminate')
    workspace_path = output_path / str(k)
    

    if noise_threshold==0:
        shutil.copy(workspace_path / train_name, workspace_path / save_name)
    else:
        if use_cuda:
            #os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
            cmd = 'python -u kge/run.py --cuda --do_eval ' \
                  '--model {} -init {} --train_path {} --noise_threshold {}  ' \
                  '--eliminate_noise_path {} --data_path {}'.format(
                model, output_path / model_name, workspace_path / train_name,
                noise_threshold, workspace_path / save_name, data_path)
        else:
            cmd = 'python -u kge/run.py --do_eval --model {} -init {} --train_path {} ' \
                  '--noise_threshold {}  --eliminate_noise_path {} --data_path {}'.format(
                model, output_path / model_name, workspace_path / train_name,
                noise_threshold, workspace_path / save_name, data_path)
        os.system(cmd)


def run_potential(workspace_path, model_path, data_path='./data/kinship',
                  train_path='./data/kinship/train.txt', rule_name='MLN_rule.txt',
                  top_k_threshold = 0.1, use_cuda=True, cuda=2):

    model = pt.SelectHiddenTriples(data_path, train_path,
                                   hidden_triples_path=workspace_path,
                                   rule_name=rule_name)

    model.run(threshold=50)  # find all hidden triple # threshold (50)：randomly sample 50 col and row

    # model.eval(use_cuda, cuda, model_path=workspace_path+'/kge_model/')
    model.eval(use_cuda, cuda, model_path=model_path, top_k_threshold=top_k_threshold)


def train(data_folder: Path, cuda, record_name: str, kge_model, iterations, noise_threshold, top_k_threshold, is_init):
    if cuda == '-1':
        use_cuda = False
    else:
        use_cuda = True
    output_path = data_folder / "record" / record_name
    check_path(output_path)

    kge_batch = 512
    kge_neg = 128
    kge_dim = 500
    kge_gamma = 6.0
    kge_alpha = 0.5
    kge_lr = 0.0005
    kge_iters = 1000
    # final_kge_iters = 80000
    kge_tbatch = 8
    kge_reg = 0.000001

    if data_folder == 'kinship':
        if kge_model == 'TransE':
            # kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, final_kge_iters, kge_tbatch, kge_reg = 1024, 256, 100, 24, 1, 0.001, 80000, 50000, 16, 0.0
            kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, kge_reg = 1024, 256, 100, 24, 1, 0.001, 50000, 16, 0.0
        if kge_model == 'DistMult':
            # kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, final_kge_iters, kge_tbatch, kge_reg = 1024, 256, 1000, 24, 1, 0.001, 50000, 50000, 16, 0.0
            kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, kge_reg = 1024, 256, 1000, 24, 1, 0.001, 50000, 16, 0.0


    check_path(output_path / "0")
    shutil.copy(data_folder / "train.txt", output_path / "0" / "fc_train.txt")

    kge_data_path = output_path / "kge_data"
    check_path(kge_data_path)

    for k in range(iterations):

        workspace_path = output_path / str(k)
        check_path(workspace_path)

        print('Start Forward Chaining...')
        run_fc(data_folder, workspace_path / 'fc_train.txt', workspace_path / 'infer.txt', 'MLN_rule.txt')
        with open(workspace_path / "fc_train.txt", "r") as fc_train_file, open(workspace_path / "infer.txt", "r") as infer_file, open(workspace_path / "observed.txt", "w") as observed_file:
                observed_file.writelines(fc_train_file.readlines())
                observed_file.writelines(infer_file.readlines())

        if k==0 and is_init==0:
            print('Start KGE Training...')
            with open(workspace_path / "fc_train.txt", "r") as fc_train_file, open(workspace_path / "infer.txt", "r") as infer_file, open(kge_data_path / "kge_train.txt", "w") as kge_train_file:
                kge_train_file.writelines(fc_train_file.readlines())
                kge_train_file.writelines(infer_file.readlines())
            run_kge(output_path, data_folder, kge_data_path / 'kge_train.txt', kge_model, 'kge_model', 
                    kge_batch=kge_batch, kge_neg=kge_neg, kge_dim=kge_dim, kge_gamma=kge_gamma, 
                    kge_alpha=kge_alpha, kge_lr=kge_lr, kge_iters=kge_iters, kge_tbatch=kge_tbatch, 
                    use_cuda=use_cuda)


        print('Start Eval and Eliminating...')
        eval_and_eliminate(output_path, kge_data_path, k, kge_model, 'kge_model', 'observed.txt', 'new_observed.txt',
                           noise_threshold=noise_threshold, use_cuda=use_cuda)

        print('Start Finding Potential Useful Triples...')
        run_potential(workspace_path=workspace_path, model_path=output_path / 'kge_model', data_path=data_folder,
                      train_path=workspace_path / 'new_observed.txt', rule_name='MLN_rule.txt', 
                      top_k_threshold = top_k_threshold,
                      use_cuda=use_cuda, cuda=cuda)

        next_workspace_path = output_path / str(k+1)
        check_path(next_workspace_path)
        with open(workspace_path / "new_observed.txt", "r") as new_observed_file, open(workspace_path / "selected_triples.txt", "r") as selected_triples_file, open(next_workspace_path / "fc_train.txt", "w") as fc_train_file:
            fc_train_file.writelines(new_observed_file.readlines())
            fc_train_file.writelines(selected_triples_file.readlines())


    workspace_path = output_path / str(iterations)
    # run_kge(path, workspace_path+'/fc_train.txt', model=kge_model, model_name='final_model',
    # 	kge_iters=kge_iters, use_cuda=use_cuda)

    final_path = output_path / "final"
    check_path(final_path)
    print('Start Forward Chaining...')
    print (data_folder / 'final_rules' / 'fc_observation.txt')
    FC1 = fc.ForwardChain(data_folder, workspace_path / 'fc_train.txt', final_path / 'inferred_obs.txt', 'final_rules/fc_observation.txt')
    FC1.run()
    FC2 = fc.ForwardChain(data_folder, workspace_path / 'fc_train.txt', final_path / 'inferred_vis.txt', 'final_rules/fc_visibility.txt')
    FC2.run()
    return final_path
