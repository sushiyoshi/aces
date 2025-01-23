import sys
sys.path.append('..')
from pyaces import *
def test_refresh():
    p = 32
    q = 32**5+1
    dim = 5
    N = 5
    ac = ArithChannel(p,q,dim,N)
    print("[*] p,q,dim,N =", p,q,dim,N)
    q_p = q/p
    print("[*] q/p =", q_p)

    f0,f1,vanmod,intmod,dim_,N_,u,tensor = ac.publish(fhe=True)
    bob = ACES(f0,f1,vanmod,intmod,dim_,N_,u)

    alice = ACESReader(ac)

    m = 4
    print("[*] original message =", m)
    ciph, noise_vector = bob.encrypt(m)
    print("[*] original ciphertext uplvl =", ciph.uplvl)
    alg = ACESAlgebra(vanmod,intmod,dim,u,tensor)
    ref = ACESRefresher(bob, alg,bob.encrypt,alice.decrypt)

    ps_c0,ps_c1 = ref.generate_pseudocipertext(ciph)
    ps_c0 = [p % intmod for p in ps_c0]
    ps_c1 = ps_c1 % intmod
    plain_before = alice.decrypt(ciph)
    print("[*] decrypt before refresh =", plain_before)
    
    # get refreshable ciphertext
    refreshable_ciph = ciph
    count = 0
    while not ref.is_refreshable(refreshable_ciph,ac.x):
        ciph_zero = bob.encrypt(0)[0]
        refreshable_ciph = alg.add(refreshable_ciph,ciph_zero)
        count += 1
    print(f"count: {count}")

    # refresh
    ref.refresh_inspector(refreshable_ciph,ac.x)

    new_ciph = ref.refresh(refreshable_ciph,ac.x)
    # refresh 後の復号
    plain_after = alice.decrypt(new_ciph)
    print("[*] decrypt after refresh =", plain_after)

    print("[*] new ciphertext uplvl =", new_ciph.uplvl)

    assert plain_after == m, "Refresh failed or message mismatch"
if __name__ == "__main__":
    test_refresh()