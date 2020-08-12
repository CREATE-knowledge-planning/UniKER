import sys
import os
import numpy as np

from fc import fc


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def run_kge(path, k, model, model_name, use_cuda=True, from_scratch=False):
    workspace_path = path + '/' + str(k) + '/'

    if not from_scratch:
        if use_cuda:
            if k > 0:
                print('run_kge with init')
                init_path = path + '/' + str(k - 1) + '/'
                cmd = 'CUDA_VISIBLE_DEVICES={} python kge/run.py --cuda --do_train --do_valid --do_test --model {} --data_path {} -b {} -n {} -d {} -g {} -a {} -adv -lr {} --max_steps {} --test_batch_size {} -save {} -init {} --train_path {}'.format(
                    cuda, model, data_path, kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters * (k + 1),
                    kge_tbatch, workspace_path + '/' + model_name, init_path + model_name, workspace_path + '/train.txt')

            else:
                cmd = 'CUDA_VISIBLE_DEVICES={} python kge/run.py --cuda --do_train --do_valid --do_test --model {} --data_path {} -b {} -n {} -d {} -g {} -a {} -adv -lr {} --max_steps {} --test_batch_size {} -save {} --train_path {}'.format(
                    cuda, model, data_path, kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch,
                    workspace_path + '/' + model_name, workspace_path + '/train.txt')
        else:
            if k > 0:
                print('run_kge with init')
                init_path = path + '/' + str(k-1) + '/'
                cmd = 'python kge/run.py --do_train --do_valid --do_test --model {} --data_path {} -b {} -n {} -d {} -g {} -a {} -adv -lr {} --max_steps {} --test_batch_size {} -save {} -init {} --train_path {}'.format(
                    model, data_path, kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters * (k+1), kge_tbatch,
                    workspace_path + '/' + model_name, init_path + model_name, workspace_path + '/train.txt')

            else:
                cmd = 'python kge/run.py --do_train --do_valid --do_test --model {} --data_path {} -b {} -n {} -d {} -g {} -a {} -adv -lr {} --max_steps {} --test_batch_size {} -save {} --train_path {}'.format(
                model, data_path, kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch,
                workspace_path + '/' + model_name, workspace_path + '/train.txt')
    else:
        if use_cuda:
            cmd = 'CUDA_VISIBLE_DEVICES={} python kge/run.py --cuda --do_train --do_valid --do_test --model {} --data_path {} -b {} -n {} -d {} -g {} -a {} -adv -lr {} --max_steps {} --test_batch_size {} -save {} --train_path {}'.format(
                cuda, model, data_path, kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, final_kge_iters,
                kge_tbatch,
                workspace_path + '/' + model_name, workspace_path + '/train.txt')
        else:
            cmd = 'python kge/run.py --do_train --do_valid --do_test --model {} --data_path {} -b {} -n {} -d {} -g {} -a {} -adv -lr {} --max_steps {} --test_batch_size {} -save {} --train_path {}'.format(
                model, data_path, kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, final_kge_iters, kge_tbatch,
                workspace_path + '/' + model_name, workspace_path + '/train.txt')

    os.system(cmd)


def run_fc(workspace_path, train_file_name, save_name, mln_rules):
    FC = fc.ForwardChain(workspace_path, os.path.join(workspace_path, train_file_name), os.path.join(workspace_path, save_name), mln_rules)
    FC.run()


def eval_and_eliminate(path, k, model, model_name, train_name, save_name, noise_threshold=0.1, use_cuda=True):
    # read train.txt, write new_train.txt
    # read infer.txt, write new_infer.txt

    print('eval_and_eliminate')
    workspace_path = path + '/' + str(k) + '/'

    if use_cuda:
        cmd = 'CUDA_VISIBLE_DEVICES={} python -u kge/run.py --cuda --do_eval --model {} -init {} --train_path {} --noise_threshold {}  --eliminate_noise_path {} --data_path {}'.format(
            cuda, model, workspace_path + '/' + model_name, workspace_path + '/' + train_name, noise_threshold,
                   workspace_path + '/' + save_name, data_path)
    else:
        cmd = 'python -u kge/run.py --do_eval --model {} -init {} --train_path {} --noise_threshold {}  --eliminate_noise_path {} --data_path {}'.format(
            model, workspace_path + '/' + model_name, workspace_path + '/' + train_name, noise_threshold,
            workspace_path + '/' + save_name, data_path)


    os.system(cmd)


if __name__ == '__main__':
    dataset = sys.argv[1]
    data_path = './data/' + sys.argv[1] + '/'

    cuda = sys.argv[2]
    if cuda == '-1':
        use_cuda = False
    else:
        use_cuda = True

    record_name = sys.argv[3]
    path = './record/' + record_name + '/'
    check_path(path)

    if len(sys.argv) == 5:
        kge_model = sys.argv[4]
    else:
        kge_model = 'TransE'

    iterations = int(sys.argv[5])

    noise_threshold = float(sys.argv[6])

    kge_batch = 512
    kge_neg = 128
    kge_dim = 500
    kge_gamma = 6.0
    kge_alpha = 0.5
    kge_lr = 0.0005
    kge_iters = 5000
    final_kge_iters = 80000
    kge_tbatch = 8
    kge_reg = 0.000001

    # data=wn18rr, TransE
    # kge_batch = 512
    # kge_neg = 1024
    # kge_dim = 1000
    # kge_gamma = 6.0
    # kge_alpha = 0.5
    # kge_lr = 0.0001
    # kge_iters = 5000
    # final_kge_iters = 80000
    # kge_tbatch = 8
    # kge_reg = 0.0

    tag = sys.argv[7]

    if dataset == 'kinship':
        if kge_model == 'TransE':
            kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, final_kge_iters, kge_tbatch, kge_reg = 1024, 256, 100, 24, 1, 0.001, 5000, 50000, 16, 0.0
        if kge_model == 'DistMult':
            kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, final_kge_iters, kge_tbatch, kge_reg = 1024, 256, 1000, 24, 1, 0.001, 5000, 50000, 16, 0.0
    elif dataset == 'noise_kinship':
        if kge_model == 'TransE':
            kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, final_kge_iters, kge_tbatch, kge_reg = 1024, 256, 100, 24, 1, 0.001, 5000, 50000, 16, 0.0
        if kge_model == 'DistMult':
            kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, final_kge_iters, kge_tbatch, kge_reg = 1024, 256, 1000, 24, 1, 0.001, 5000, 50000, 16, 0.0
    elif dataset == 'FB15k-237':
        if kge_model == 'TransE':
            kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, final_kge_iters, kge_tbatch, kge_reg = 1024, 256, 1000, 9.0, 1, 0.001, 5000, 100000, 16, 0.00005
        if kge_model == 'DistMult':
            kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, final_kge_iters, kge_tbatch, kge_reg = 1024, 256, 2000, 200.0, 1, 0.001, 5000, 100000, 16, 0.00001
    elif dataset == 'wn18rr':
        if kge_model == 'TransE':
            if tag == '0':
                kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, final_kge_iters, kge_tbatch, kge_reg = 512, 1024, 1000, 6.0, 0.5, 0.00001, 5000, 80000, 8, 0.0
            elif tag == '1':
                kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, final_kge_iters, kge_tbatch, kge_reg = 512, 1024, 1000, 6.0, 0.5, 0.00005, 5000, 80000, 8, 0.0
            elif tag == '2':
                kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, final_kge_iters, kge_tbatch, kge_reg = 512, 1024, 1000, 6.0, 0.5, 0.0001, 5000, 80000, 8, 0.0
            elif tag == '3':
                kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, final_kge_iters, kge_tbatch, kge_reg = 512, 1024, 1000, 6.0, 0.5, 0.00005, 5000, 80000, 8, 0.000005
            elif tag == '4':
                kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, final_kge_iters, kge_tbatch, kge_reg = 512, 1024, 1000, 6.0, 0.5, 0.0001, 5000, 80000, 8, 0.000005

        if kge_model == 'DistMult':
            if tag == '0':
                kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, final_kge_iters, kge_tbatch, kge_reg = 512, 1024, 1000, 200.0, 1, 0.002, 5000, 90000, 8, 0.000005
            elif tag == '1':
                kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, final_kge_iters, kge_tbatch, kge_reg = 512, 1024, 2000, 200.0, 1, 0.002, 5000, 90000, 8, 0.000005
            elif tag == '2':
                kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, final_kge_iters, kge_tbatch, kge_reg = 512, 1024, 1000, 200.0, 1, 0.001, 5000, 90000, 8, 0.000005
            elif tag == '3':
                kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, final_kge_iters, kge_tbatch, kge_reg = 512, 1024, 2000, 200.0, 1, 0.001, 5000, 90000, 8, 0.000005

    check_path(path + '/0/')
    os.system('cp {}/train.txt {}/train.txt'.format(data_path, path+'/0/'))

    for k in range(iterations):

        workspace_path = path + '/' + str(k) + '/'
        check_path(workspace_path)

        run_kge(path, k, kge_model, 'mymodel', use_cuda=use_cuda)
        eval_and_eliminate(path, k, kge_model, 'mymodel', 'train.txt', 'new_train.txt', noise_threshold=noise_threshold, use_cuda=use_cuda)

        if k == 0:
            os.system('cp {}/train.txt {}/fc_train.txt'.format(data_path, workspace_path))
        else:
            prev_workspace_path = path + '/' + str(k - 1) + '/'
            os.system(
                'cat {}/fc_train.txt {}/new_train.txt >> {}/fc_train.txt'.format(prev_workspace_path, workspace_path,
                                                                                 workspace_path))
        print('Start Foward Chaining...')
        run_fc(workspace_path, 'fc_train.txt', 'infer.txt')
        eval_and_eliminate(path, k, kge_model, 'mymodel', 'infer.txt', 'new_infer.txt', noise_threshold=noise_threshold, use_cuda=use_cuda)

        check_path(path + '/' + str(k+1) + '/')
        os.system('cp {}/new_infer.txt  {}/train.txt'.format(workspace_path, path + '/' + str(k + 1) + '/'))
    workspace_path = path + '/' + str(iterations) + '/'
    for k in range(iterations):
        tmp_path = path + '/' + str(k) + '/'
        os.system('cat {}/train.txt {}/new_train.txt >> {}/train.txt'.format(workspace_path, tmp_path, workspace_path))
    # if dataset == 'wn18rr':
    #     tmp_path = path + '/0/'
    #     os.system('cat {}/train.txt {}/new_train.txt >> {}/train.txt'.format(workspace_path, tmp_path, workspace_path))

    run_kge(path, iterations, kge_model, 'final_model', use_cuda=use_cuda, from_scratch=True)

