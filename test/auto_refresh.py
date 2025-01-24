############################################
# test2_auto_compare.py
#  - test2.py と同じ refresh のやり方(kの求め方)で、
#    自動refreshあり/なしの2通りを比較し、精度を出力する例
############################################

import sys
sys.path.append('..')  # pyaces へのパス

import random as rd
from pyaces import *

def main():
    # パラメータ
    vanmod = 32
    intmod = 32**6 + 1
    dim = 3
    N = 3

    # チャネル初期化
    ac = ArithChannel(vanmod,intmod,dim,N)
    print(f"vanmod={ac.vanmod}, intmod={ac.intmod}, dim={ac.dim}, N={ac.N}")

    # fhe=True => tensor など追加情報が返る -> refresh 可能
    (f0,f1,vanmod,intmod,dim,N,u,tensor) = ac.publish(fhe = True)
    bob = ACES(f0,f1,vanmod,intmod,dim,N,u)
    alice = ACESReader(ac)
    alg = ACESAlgebra(vanmod,intmod,dim,u,tensor)
    rfr = ACESRefresher(ac,alg,bob.encrypt,alice.decrypt)

    q_p = intmod / vanmod
    print(f"q_p={q_p}")

    #==== テストする式(命令列) ====
    # 質問文の test2.py での例：
    # true_fun = Algebra().compile("(0*1+2*3+4*5)*6+7")
    # 今回はこの式で試す
    # expr_str = "(0*1+2*3+4*5)*6+7"
    expr_str = "(0*1+2*3)"

    #=== 関数にパース(平文で計算する用) ===
    true_fun = Algebra().compile(expr_str)  


    #=== 同型演算(暗号文)で使う関数 ===
    send_fun = alg.compile(expr_str)  
    keep_fun = rfr.compile(expr_str)  

    def compute_with_auto_refresh(send_array, keep_array, do_auto_refresh,ground_val):
        subexpr1 = "(0*1+2*3)"
        # subexpr1 = "(0*1+2*3+4*5)*6+7"

        fun1_send = alg.compile(subexpr1)
        c1 = fun1_send(send_array)  # これが online1 相当

        # refresh したいときの "keep_fun1"
        fun1_keep = rfr.compile(subexpr1)
        do_auto_refresh = do_auto_refresh

        # do_auto_refresh なら refresh の k を求めて適用
        if do_auto_refresh:
            prev_uplvl = c1.uplvl
            c1 = rfr.refresh(c1,ac.x)
            d = alice.decrypt(c1)
            prev_uplvl = c1.uplvl
            while c1.uplvl > q_p:
                c1 = rfr.refresh(c1,ac.x)
                d = alice.decrypt(c1)
                print(f"Refreshed: {prev_uplvl} -> {c1.uplvl}, uplvl-q_p={c1.uplvl-q_p}, dec={alice.decrypt(c1)}")
                prev_uplvl = c1.uplvl
            if d == ground_val:
                # 緑
                print("\033[32m", end="")
            else:
                # 赤
                print("\033[31m", end="")
            print(f"uplvl-q_p={c1.uplvl-q_p}, dec={alice.decrypt(c1)}")
            print("\033[0m", end="")
            
        return c1

    #==== 以下で「自動refreshあり」「なし」を比較し、正答率を出す ====
    # ランダムテスト回数
    num_tests = 10

    # 正解数
    correct_no_refresh = 0
    correct_with_refresh = 0

    for i in range(num_tests):
        print(f"--------------------Test {i+1}/{num_tests}")
        # ランダムに 8個の要素を生成
        array = [rd.randint(0,5) for _ in range(8)]
        # 暗号化
        enc_array = [bob.encrypt(a) for a in array]
        send_array, keep_array = map(list,zip(*enc_array))
        print(f"{array[0]}*{array[1]}+{array[2]}*{array[3]}+{array[4]}*{array[5]}={array[0]*array[1]+array[2]*array[3]+array[4]*array[5]}")
        # --- GroundTruth (平文計算)
        ground_val = true_fun(array) % vanmod
        print(f"GroundTruth: {ground_val}")

        # --- (A) refreshなし計算
        c_no_ref = compute_with_auto_refresh(send_array, keep_array, do_auto_refresh=False,ground_val=ground_val)
        dec_no_ref = alice.decrypt(c_no_ref) % vanmod
        if dec_no_ref == ground_val:
            correct_no_refresh += 1
            # 緑にする
            print("\033[32m", end="")
        else:
            # 赤くする
            print("\033[31m", end="")
        print(f"Refreshなし: dec={dec_no_ref}, uplvl={c_no_ref.uplvl}")
        print("\033[0m", end="")

        # --- (B) refreshあり計算
        c_ref = compute_with_auto_refresh(send_array, keep_array, do_auto_refresh=True,ground_val=ground_val)
        dec_ref = alice.decrypt(c_ref) % vanmod
        if dec_ref == ground_val:
            correct_with_refresh += 1
            # 緑にする
            print("\033[32m", end="")
        else:
            # 赤くする
            print("\033[31m", end="")
        print(f"Refreshあり: dec={dec_ref}, uplvl={c_ref.uplvl}")
        #戻す
        print("\033[0m", end="")

    #=== 精度(正答率)を表示 ===
    accuracy_no_refresh = correct_no_refresh / num_tests
    accuracy_with_refresh = correct_with_refresh / num_tests
    print("=========================================")
    print(f"総テスト数: {num_tests}")
    print(f" - Refreshしない場合の正答率: {accuracy_no_refresh:.3f}")
    print(f" - Refreshする場合の正答率:   {accuracy_with_refresh:.3f}")

if __name__ == "__main__":
    main()
