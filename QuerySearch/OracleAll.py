from rouge import Rouge
import copy
import json
import random
random.seed(413)
rouge = Rouge()

def get_score(hyp,ref):
    try:
        temp_rouge = rouge.get_scores(hyp, ref)
        cur_score = (temp_rouge[0]["rouge-1"]['f'] + temp_rouge[0]["rouge-2"]['f'] + temp_rouge[0]["rouge-l"]['f'])/3
    except :
        cur_score = 0
    return cur_score

def get_oracle(sent_list,summary):
    Chosen_idx = []
    best_score = 0
    cal_count = 0 
    while 1:
        best_choice = -1
        best_sub_score = 0
        for i in range(len(sent_list)):
            if i not in Chosen_idx and len(sent_list[i]) != 0 :
                cal_count += 1
                temp_chosen = copy.deepcopy(Chosen_idx)
                temp_chosen.append(i)
                temp_chosen_sents = [sent_list[i] for i in temp_chosen]
                #print(temp_chosen)
                #print(temp_chosen_sents)
                cur_score = get_score(" ".join(temp_chosen_sents),summary)
                cur_sub_score = cur_score - best_score
                if cur_sub_score > best_sub_score:
                    best_sub_score = cur_sub_score
                    best_choice = i
        if best_choice == -1:
            break
        Chosen_idx.append(best_choice)
        best_sents = [sent_list[i] for i in Chosen_idx]
        best_score = get_score(" ".join(best_sents),summary)

    best_sents = [sent_list[i] for i in Chosen_idx]
    #print(len(sent_list))
    #print(len(best_sents))
    #print(cal_count)
    try:
        temp_rouge = rouge.get_scores(" ".join(best_sents), summary)
    except :
        return 0,0,0
    
    return temp_rouge[0]["rouge-1"]['f'],temp_rouge[0]["rouge-2"]['f'],temp_rouge[0]["rouge-l"]['f']

f = open('test.extract.source','r',encoding = 'utf-8')
f2 = open('test.target','r',encoding = 'utf-8')
f3 = open('QueryResult.txt','r',encoding = 'utf-8')
query = f3.readlines()
query = [[int (j) for j in i.strip().split()] for i in query]


summarys = f2.readlines()
summarys = [i.strip() for i in summarys]
import random
data_index = []
while len(data_index) < 2000:
    random_index = random.randint(0,len(summarys)-1)
    if random_index not in data_index:
        data_index.append(random_index)

print(data_index[:10])
assert data_index[0] == 10455


r1 = 0 
r2 = 0
rl = 0
lines = f.readlines()

ftrain = open('train.extract.source','r',encoding = 'utf-8')
assist_lines = ftrain.readlines()

from tqdm import tqdm 
for i in tqdm(range(len(lines))):
    data = lines[i].strip()
    data_dict = json.loads(data)
    doc = data_dict['text']

    for j in query[i][:1]:
        assist = assist_lines[j].strip()
        assist_dict = json.loads(assist)
        assist_doc = assist_dict['text']
        doc = assist_doc + doc
   # doc = [i.replace("\\","") for i in doc]
    r1_s,r2_s,rl_s = get_oracle(doc,summarys[data_index[i]])
    r1 += r1_s 
    r2 += r2_s 
    rl += rl_s

print("ROUGE Score : ROUGE1: {} ROUGE2: {} ROUGEL: {}".format(r1/len(lines), r2/len(lines), rl/len(lines)))

[10455, 7668, 2977, 1326, 9000, 4793, 2898, 10761, 9206, 2304]
本地计算机有: 8 核心
本地计算机有: 8 核心
本地计算机有: 8 核心
Traceback (most recent call last):
  File "<string>", line 1, in <module>
本地计算机有: 8 核心
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 116, in spawn_main
    exitcode = _main(fd, parent_sentinel)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 125, in _main
    prepare(preparation_data)
Traceback (most recent call last):
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 236, in prepare
    _fixup_main_from_path(data['init_main_from_path'])
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 287, in _fixup_main_from_path
    main_content = runpy.run_path(main_path,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 268, in run_path
  File "<string>", line 1, in <module>
本地计算机有: 8 核心
    return _run_module_code(code, init_globals, run_name,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 97, in _run_module_code
本地计算机有: 8 核心
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 116, in spawn_main
    _run_code(code, mod_globals, init_globals,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 148, in <module>
    multi_process_tag(data)
    exitcode = _main(fd, parent_sentinel)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 125, in _main
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 74, in multi_process_tag
    pool = mp.Pool(num_cores)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 119, in Pool
    return Pool(processes, initializer, initargs, maxtasksperchild,
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 212, in __init__
    prepare(preparation_data)
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 236, in prepare
    self._repopulate_pool()
本地计算机有: 8 核心
    _fixup_main_from_path(data['init_main_from_path'])
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 287, in _fixup_main_from_path
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 303, in _repopulate_pool
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 116, in spawn_main
    exitcode = _main(fd, parent_sentinel)
    return self._repopulate_pool_static(self._ctx, self.Process,
本地计算机有: 8 核心
    main_content = runpy.run_path(main_path,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 268, in run_path
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 326, in _repopulate_pool_static
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 125, in _main
    return _run_module_code(code, init_globals, run_name,
    w.start()
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 97, in _run_module_code
  File "D:\App\miniconda\envs\main\lib\multiprocessing\process.py", line 121, in start
    prepare(preparation_data)
    _run_code(code, mod_globals, init_globals,
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 116, in spawn_main
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 236, in prepare
    exitcode = _main(fd, parent_sentinel)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 125, in _main
    prepare(preparation_data)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 236, in prepare
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 87, in _run_code
    _fixup_main_from_path(data['init_main_from_path'])
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    exec(code, run_globals)
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 148, in <module>
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 287, in _fixup_main_from_path
Traceback (most recent call last):
    self._popen = self._Popen(self)
    _fixup_main_from_path(data['init_main_from_path'])
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 116, in spawn_main
    exitcode = _main(fd, parent_sentinel)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 125, in _main
  File "<string>", line 1, in <module>
    multi_process_tag(data)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 327, in _Popen
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 287, in _fixup_main_from_path
    prepare(preparation_data)
    main_content = runpy.run_path(main_path,
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 74, in multi_process_tag
    main_content = runpy.run_path(main_path,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 268, in run_path
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 236, in prepare
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 268, in run_path
    return _run_module_code(code, init_globals, run_name,
    pool = mp.Pool(num_cores)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 119, in Pool
    return _run_module_code(code, init_globals, run_name,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 97, in _run_module_code
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 97, in _run_module_code
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 116, in spawn_main
    return Popen(process_obj)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\popen_spawn_win32.py", line 45, in __init__
    return Pool(processes, initializer, initargs, maxtasksperchild,
    _run_code(code, mod_globals, init_globals,
    _fixup_main_from_path(data['init_main_from_path'])
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 287, in _fixup_main_from_path
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 212, in __init__
    _run_code(code, mod_globals, init_globals,
    self._repopulate_pool()
    exitcode = _main(fd, parent_sentinel)
    prep_data = spawn.get_preparation_data(process_obj._name)
    main_content = runpy.run_path(main_path,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 268, in run_path
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 87, in _run_code
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 303, in _repopulate_pool
    return _run_module_code(code, init_globals, run_name,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 97, in _run_module_code
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 87, in _run_code
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 125, in _main
    exec(code, run_globals)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 154, in get_preparation_data
    return self._repopulate_pool_static(self._ctx, self.Process,
    exec(code, run_globals)
    prepare(preparation_data)
    _check_not_importing_main()
    _run_code(code, mod_globals, init_globals,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 87, in _run_code
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 326, in _repopulate_pool_static
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 148, in <module>
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 236, in prepare
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 134, in _check_not_importing_main
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 148, in <module>
    exec(code, run_globals)
    w.start()
    multi_process_tag(data)
    _fixup_main_from_path(data['init_main_from_path'])
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 148, in <module>
    raise RuntimeError('''
  File "D:\App\miniconda\envs\main\lib\multiprocessing\process.py", line 121, in start
    multi_process_tag(data)
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 74, in multi_process_tag
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 287, in _fixup_main_from_path
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.    multi_process_tag(data)
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 74, in multi_process_tag
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 74, in multi_process_tag
    pool = mp.Pool(num_cores)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 119, in Pool
    main_content = runpy.run_path(main_path,

    self._popen = self._Popen(self)
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    pool = mp.Pool(num_cores)
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 268, in run_path
    return Pool(processes, initializer, initargs, maxtasksperchild,
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 327, in _Popen
    pool = mp.Pool(num_cores)
Traceback (most recent call last):
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 119, in Pool
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 212, in __init__
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 119, in Pool
    return _run_module_code(code, init_globals, run_name,
  File "<string>", line 1, in <module>
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 116, in spawn_main
    self._repopulate_pool()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 303, in _repopulate_pool
    return Popen(process_obj)
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 97, in _run_module_code
    return Pool(processes, initializer, initargs, maxtasksperchild,
    return Pool(processes, initializer, initargs, maxtasksperchild,
    exitcode = _main(fd, parent_sentinel)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\popen_spawn_win32.py", line 45, in __init__
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 116, in spawn_main
    return self._repopulate_pool_static(self._ctx, self.Process,
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 212, in __init__
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 212, in __init__
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 125, in _main
    _run_code(code, mod_globals, init_globals,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 87, in _run_code
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 326, in _repopulate_pool_static
    exitcode = _main(fd, parent_sentinel)
    self._repopulate_pool()
    self._repopulate_pool()
    prepare(preparation_data)
    prep_data = spawn.get_preparation_data(process_obj._name)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 125, in _main
    exec(code, run_globals)
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 148, in <module>
    prepare(preparation_data)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 303, in _repopulate_pool
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 236, in prepare
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 154, in get_preparation_data
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 303, in _repopulate_pool
    _fixup_main_from_path(data['init_main_from_path'])
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 236, in prepare
    multi_process_tag(data)
    return self._repopulate_pool_static(self._ctx, self.Process,
    _check_not_importing_main()
    _fixup_main_from_path(data['init_main_from_path'])
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 287, in _fixup_main_from_path
    return self._repopulate_pool_static(self._ctx, self.Process,
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 74, in multi_process_tag
    main_content = runpy.run_path(main_path,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 268, in run_path
    w.start()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\process.py", line 121, in start
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 326, in _repopulate_pool_static
    self._popen = self._Popen(self)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 327, in _Popen
    pool = mp.Pool(num_cores)
    return _run_module_code(code, init_globals, run_name,
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 287, in _fixup_main_from_path
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 326, in _repopulate_pool_static
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 134, in _check_not_importing_main
    w.start()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 119, in Pool
    return Popen(process_obj)
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 97, in _run_module_code
    w.start()
    main_content = runpy.run_path(main_path,
  File "D:\App\miniconda\envs\main\lib\multiprocessing\process.py", line 121, in start
    raise RuntimeError('''
  File "D:\App\miniconda\envs\main\lib\multiprocessing\popen_spawn_win32.py", line 45, in __init__
    return Pool(processes, initializer, initargs, maxtasksperchild,
  File "D:\App\miniconda\envs\main\lib\multiprocessing\process.py", line 121, in start
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 268, in run_path
    _run_code(code, mod_globals, init_globals,
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.    self._popen = self._Popen(self)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 327, in _Popen
    prep_data = spawn.get_preparation_data(process_obj._name)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 154, in get_preparation_data
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 87, in _run_code
    return _run_module_code(code, init_globals, run_name,

  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 212, in __init__
    self._popen = self._Popen(self)
    return Popen(process_obj)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\popen_spawn_win32.py", line 45, in __init__
    exec(code, run_globals)
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 148, in <module>
    self._repopulate_pool()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 303, in _repopulate_pool
    _check_not_importing_main()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 134, in _check_not_importing_main
    return self._repopulate_pool_static(self._ctx, self.Process,
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 327, in _Popen
    multi_process_tag(data)
    prep_data = spawn.get_preparation_data(process_obj._name)
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 97, in _run_module_code
    return Popen(process_obj)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\popen_spawn_win32.py", line 45, in __init__
    _run_code(code, mod_globals, init_globals,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 87, in _run_code
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 326, in _repopulate_pool_static
    raise RuntimeError('''
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
    prep_data = spawn.get_preparation_data(process_obj._name)
    exec(code, run_globals)
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 148, in <module>
    w.start()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\process.py", line 121, in start
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 154, in get_preparation_data
    self._popen = self._Popen(self)
    multi_process_tag(data)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 154, in get_preparation_data
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 74, in multi_process_tag
    _check_not_importing_main()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 134, in _check_not_importing_main
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 74, in multi_process_tag
    _check_not_importing_main()
    raise RuntimeError('''
    pool = mp.Pool(num_cores)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 134, in _check_not_importing_main
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 327, in _Popen
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
    return Popen(process_obj)
    raise RuntimeError('''
    pool = mp.Pool(num_cores)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 119, in Pool
  File "D:\App\miniconda\envs\main\lib\multiprocessing\popen_spawn_win32.py", line 45, in __init__
    return Pool(processes, initializer, initargs, maxtasksperchild,
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 212, in __init__
    prep_data = spawn.get_preparation_data(process_obj._name)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 154, in get_preparation_data
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
    self._repopulate_pool()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 303, in _repopulate_pool
    return self._repopulate_pool_static(self._ctx, self.Process,
    _check_not_importing_main()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 134, in _check_not_importing_main
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 326, in _repopulate_pool_static
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 119, in Pool
    raise RuntimeError('''
RuntimeError: 
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
    w.start()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\process.py", line 121, in start
    return Pool(processes, initializer, initargs, maxtasksperchild,
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 212, in __init__
    self._repopulate_pool()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 303, in _repopulate_pool
    self._popen = self._Popen(self)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 327, in _Popen
    return Popen(process_obj)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\popen_spawn_win32.py", line 45, in __init__
    prep_data = spawn.get_preparation_data(process_obj._name)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 154, in get_preparation_data
    _check_not_importing_main()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 134, in _check_not_importing_main
    raise RuntimeError('''
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
    return self._repopulate_pool_static(self._ctx, self.Process,
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 326, in _repopulate_pool_static
    w.start()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\process.py", line 121, in start
    self._popen = self._Popen(self)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 327, in _Popen
    return Popen(process_obj)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\popen_spawn_win32.py", line 45, in __init__
    prep_data = spawn.get_preparation_data(process_obj._name)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 154, in get_preparation_data
    _check_not_importing_main()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 134, in _check_not_importing_main
    raise RuntimeError('''
RuntimeError: 
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
[10455, 7668, 2977, 1326, 9000, 4793, 2898, 10761, 9206, 2304]
[10455, 7668, 2977, 1326, 9000, 4793, 2898, 10761, 9206, 2304]
[10455, 7668, 2977, 1326, 9000, 4793, 2898, 10761, 9206, 2304]
[10455, 7668, 2977, 1326, 9000, 4793, 2898, 10761, 9206, 2304]
[10455, 7668, 2977, 1326, 9000, 4793, 2898, 10761, 9206, 2304]
[10455, 7668, 2977, 1326, 9000, 4793, 2898, 10761, 9206, 2304]
[10455, 7668, 2977, 1326, 9000, 4793, 2898, 10761, 9206, 2304]
[10455, 7668, 2977, 1326, 9000, 4793, 2898, 10761, 9206, 2304]
本地计算机有: 8 核心
本地计算机有: 8 核心
本地计算机有: 8 核心
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 116, in spawn_main
    exitcode = _main(fd, parent_sentinel)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 125, in _main
Traceback (most recent call last):
  File "<string>", line 1, in <module>
Traceback (most recent call last):
    prepare(preparation_data)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 236, in prepare
    _fixup_main_from_path(data['init_main_from_path'])
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 287, in _fixup_main_from_path
    main_content = runpy.run_path(main_path,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 268, in run_path
本地计算机有: 8 核心
  File "<string>", line 1, in <module>
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 116, in spawn_main
    exitcode = _main(fd, parent_sentinel)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 125, in _main
    prepare(preparation_data)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 236, in prepare
    return _run_module_code(code, init_globals, run_name,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 97, in _run_module_code
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 116, in spawn_main
    _run_code(code, mod_globals, init_globals,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 87, in _run_code
    exitcode = _main(fd, parent_sentinel)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 125, in _main
本地计算机有: 8 核心
    exec(code, run_globals)
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 148, in <module>
    multi_process_tag(data)
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 74, in multi_process_tag
本地计算机有: 8 核心
    _fixup_main_from_path(data['init_main_from_path'])
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 287, in _fixup_main_from_path
    main_content = runpy.run_path(main_path,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 268, in run_path
    return _run_module_code(code, init_globals, run_name,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 97, in _run_module_code
    prepare(preparation_data)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 236, in prepare
    _run_code(code, mod_globals, init_globals,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 148, in <module>
    pool = mp.Pool(num_cores)
    multi_process_tag(data)
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 74, in multi_process_tag
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 119, in Pool
本地计算机有: 8 核心
    _fixup_main_from_path(data['init_main_from_path'])
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 287, in _fixup_main_from_path
    return Pool(processes, initializer, initargs, maxtasksperchild,
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 212, in __init__
    self._repopulate_pool()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 303, in _repopulate_pool
    return self._repopulate_pool_static(self._ctx, self.Process,
    pool = mp.Pool(num_cores)
    main_content = runpy.run_path(main_path,
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 268, in run_path
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 116, in spawn_main
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 326, in _repopulate_pool_static
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 119, in Pool
    exitcode = _main(fd, parent_sentinel)
    return Pool(processes, initializer, initargs, maxtasksperchild,
    w.start()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\process.py", line 121, in start
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 125, in _main
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 212, in __init__
    prepare(preparation_data)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 236, in prepare
    self._repopulate_pool()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 303, in _repopulate_pool
    return _run_module_code(code, init_globals, run_name,
    return self._repopulate_pool_static(self._ctx, self.Process,
    _fixup_main_from_path(data['init_main_from_path'])
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    self._popen = self._Popen(self)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 327, in _Popen
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 287, in _fixup_main_from_path
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 97, in _run_module_code
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 326, in _repopulate_pool_static
    return Popen(process_obj)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 116, in spawn_main
本地计算机有: 8 核心
    _run_code(code, mod_globals, init_globals,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 87, in _run_code
    w.start()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\process.py", line 121, in start
    exitcode = _main(fd, parent_sentinel)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 125, in _main
    main_content = runpy.run_path(main_path,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 268, in run_path
  File "D:\App\miniconda\envs\main\lib\multiprocessing\popen_spawn_win32.py", line 45, in __init__
    return _run_module_code(code, init_globals, run_name,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 97, in _run_module_code
    exec(code, run_globals)
    _run_code(code, mod_globals, init_globals,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 87, in _run_code
    prepare(preparation_data)
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 148, in <module>
    self._popen = self._Popen(self)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 327, in _Popen
    exec(code, run_globals)
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 148, in <module>
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 236, in prepare
    multi_process_tag(data)
    multi_process_tag(data)
    prep_data = spawn.get_preparation_data(process_obj._name)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 154, in get_preparation_data
    _check_not_importing_main()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 134, in _check_not_importing_main
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 74, in multi_process_tag
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 74, in multi_process_tag
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 116, in spawn_main
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    return Popen(process_obj)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\popen_spawn_win32.py", line 45, in __init__
    raise RuntimeError('''
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
    exitcode = _main(fd, parent_sentinel)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 125, in _main
    prep_data = spawn.get_preparation_data(process_obj._name)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 154, in get_preparation_data
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 116, in spawn_main
    _fixup_main_from_path(data['init_main_from_path'])
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 287, in _fixup_main_from_path
    pool = mp.Pool(num_cores)
    prepare(preparation_data)
    _check_not_importing_main()
    pool = mp.Pool(num_cores)
    exitcode = _main(fd, parent_sentinel)
    main_content = runpy.run_path(main_path,
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 236, in prepare
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 134, in _check_not_importing_main
    _fixup_main_from_path(data['init_main_from_path'])
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 287, in _fixup_main_from_path
    raise RuntimeError('''
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 268, in run_path
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 125, in _main
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 119, in Pool
    main_content = runpy.run_path(main_path,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 268, in run_path
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 119, in Pool
    return _run_module_code(code, init_globals, run_name,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 97, in _run_module_code
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 116, in spawn_main
    _run_code(code, mod_globals, init_globals,
    return _run_module_code(code, init_globals, run_name,
    prepare(preparation_data)
    return Pool(processes, initializer, initargs, maxtasksperchild,
    return Pool(processes, initializer, initargs, maxtasksperchild,
    exitcode = _main(fd, parent_sentinel)
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 87, in _run_code
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 97, in _run_module_code
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 236, in prepare
    _run_code(code, mod_globals, init_globals,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 87, in _run_code
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 125, in _main
    exec(code, run_globals)
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 148, in <module>
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 212, in __init__
    _fixup_main_from_path(data['init_main_from_path'])
    prepare(preparation_data)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 236, in prepare
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 212, in __init__
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 287, in _fixup_main_from_path
    multi_process_tag(data)
    self._repopulate_pool()
    exec(code, run_globals)
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 148, in <module>
    self._repopulate_pool()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 303, in _repopulate_pool
    main_content = runpy.run_path(main_path,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 268, in run_path
    _fixup_main_from_path(data['init_main_from_path'])
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 287, in _fixup_main_from_path
    return _run_module_code(code, init_globals, run_name,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 97, in _run_module_code
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 303, in _repopulate_pool
    multi_process_tag(data)
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 74, in multi_process_tag
    return self._repopulate_pool_static(self._ctx, self.Process,
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 326, in _repopulate_pool_static
    _run_code(code, mod_globals, init_globals,
    return self._repopulate_pool_static(self._ctx, self.Process,
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 326, in _repopulate_pool_static
    pool = mp.Pool(num_cores)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 119, in Pool
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 87, in _run_code
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 74, in multi_process_tag
    w.start()
    pool = mp.Pool(num_cores)
    w.start()
    return Pool(processes, initializer, initargs, maxtasksperchild,
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 212, in __init__
  File "D:\App\miniconda\envs\main\lib\multiprocessing\process.py", line 121, in start
    exec(code, run_globals)
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 148, in <module>
  File "D:\App\miniconda\envs\main\lib\multiprocessing\process.py", line 121, in start
    main_content = runpy.run_path(main_path,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 268, in run_path
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 119, in Pool
    return Pool(processes, initializer, initargs, maxtasksperchild,
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 212, in __init__
    self._repopulate_pool()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 303, in _repopulate_pool
    return _run_module_code(code, init_globals, run_name,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 97, in _run_module_code
    self._popen = self._Popen(self)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 327, in _Popen
    multi_process_tag(data)
    return Popen(process_obj)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\popen_spawn_win32.py", line 45, in __init__
    _run_code(code, mod_globals, init_globals,
  File "D:\App\miniconda\envs\main\lib\runpy.py", line 87, in _run_code
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 74, in multi_process_tag
    return self._repopulate_pool_static(self._ctx, self.Process,
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 326, in _repopulate_pool_static
    self._repopulate_pool()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 303, in _repopulate_pool
    w.start()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\process.py", line 121, in start
    pool = mp.Pool(num_cores)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 119, in Pool
    exec(code, run_globals)
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 148, in <module>
    return self._repopulate_pool_static(self._ctx, self.Process,
    prep_data = spawn.get_preparation_data(process_obj._name)
    self._popen = self._Popen(self)
    return Pool(processes, initializer, initargs, maxtasksperchild,
    self._popen = self._Popen(self)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 326, in _repopulate_pool_static
    multi_process_tag(data)
  File "D:\KevinproPython\workspace\Kevinpro-NLP-demo\QuerySearch\MultiTest.py", line 74, in multi_process_tag
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 327, in _Popen
    pool = mp.Pool(num_cores)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 327, in _Popen
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 154, in get_preparation_data
    w.start()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 212, in __init__
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 119, in Pool
    return Popen(process_obj)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\popen_spawn_win32.py", line 45, in __init__
  File "D:\App\miniconda\envs\main\lib\multiprocessing\process.py", line 121, in start
    return Popen(process_obj)
    self._repopulate_pool()
    _check_not_importing_main()
    return Pool(processes, initializer, initargs, maxtasksperchild,
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 212, in __init__
  File "D:\App\miniconda\envs\main\lib\multiprocessing\popen_spawn_win32.py", line 45, in __init__
    self._popen = self._Popen(self)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 303, in _repopulate_pool
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 134, in _check_not_importing_main
    prep_data = spawn.get_preparation_data(process_obj._name)
    self._repopulate_pool()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 327, in _Popen
    prep_data = spawn.get_preparation_data(process_obj._name)
    return self._repopulate_pool_static(self._ctx, self.Process,
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 154, in get_preparation_data
    raise RuntimeError('''
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 303, in _repopulate_pool
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 154, in get_preparation_data
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 326, in _repopulate_pool_static
    return Popen(process_obj)
    return self._repopulate_pool_static(self._ctx, self.Process,
  File "D:\App\miniconda\envs\main\lib\multiprocessing\pool.py", line 326, in _repopulate_pool_static
    _check_not_importing_main()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 134, in _check_not_importing_main
    w.start()
    _check_not_importing_main()
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.  File "D:\App\miniconda\envs\main\lib\multiprocessing\popen_spawn_win32.py", line 45, in __init__
    w.start()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\process.py", line 121, in start
    prep_data = spawn.get_preparation_data(process_obj._name)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 154, in get_preparation_data
    _check_not_importing_main()
    raise RuntimeError('''
  File "D:\App\miniconda\envs\main\lib\multiprocessing\process.py", line 121, in start
    self._popen = self._Popen(self)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 327, in _Popen

  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 134, in _check_not_importing_main
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
    self._popen = self._Popen(self)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\context.py", line 327, in _Popen
    return Popen(process_obj)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 134, in _check_not_importing_main
    raise RuntimeError('''
    return Popen(process_obj)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\popen_spawn_win32.py", line 45, in __init__
    raise RuntimeError('''
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
    prep_data = spawn.get_preparation_data(process_obj._name)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 154, in get_preparation_data
  File "D:\App\miniconda\envs\main\lib\multiprocessing\popen_spawn_win32.py", line 45, in __init__
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
    prep_data = spawn.get_preparation_data(process_obj._name)
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 154, in get_preparation_data
    _check_not_importing_main()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 134, in _check_not_importing_main
    raise RuntimeError('''
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
    _check_not_importing_main()
  File "D:\App\miniconda\envs\main\lib\multiprocessing\spawn.py", line 134, in _check_not_importing_main
    raise RuntimeError('''
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line 