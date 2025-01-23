import random
import math

def degree(p):
  for i,c in enumerate(p.coefs[::-1]):
    if c != 0:
      return len(p.coefs)-i-1
  return 0

class Polynomial(object):
  
  def __init__(self,coefs,intmod=None):
    self.coefs = coefs
    self.intmod = intmod

  def __repr__(self):
    return "+".join([f"[{c}]^{k}" for k,c in enumerate(self.coefs) if k <= degree(self) and c != 0][::-1])+f" ({self.intmod})"

  def mod(self,intmod=None):
    if intmod == None:
      return Polynomial(self.coefs,None)
    else:
      return Polynomial([(x % intmod) for x in self.coefs],intmod)

  def __mul__(self,other):
    mod = None if self.intmod != other.intmod else self.intmod
    mod_coef = lambda x,y: (x * y) % mod if mod != None else x*y
    d_self = degree(self)
    d_other = degree(other)
    coefs = [sum([(mod_coef(c,other.coefs[n-k]) if 0 <= n-k < len(other.coefs) else 0) for k,c in enumerate(self.coefs)]) for n in range(d_self+d_other+1)]
    return Polynomial(coefs,mod)

  def __add__(self,other):
    mod = None if self.intmod != other.intmod else self.intmod
    mod_coef = lambda x,y: (x + y) % mod if mod != None else x + y
    d_self = degree(self)
    d_other = degree(other)
    coefs = [ mod_coef(self.coefs[k] if 0<=k<len(self.coefs) else 0, other.coefs[k] if 0 <=k<len(other.coefs) else 0) for k in range(max(d_self,d_other)+1)]
    return Polynomial(coefs,mod)
  
  def __lshift__(self,other):
    d_other = degree(other)
    if other.coefs[d_other] != 1:
      print("Warning for Polynomial.__lshift__: polynomial modulus is not monic (no action taken)")
      return self, False

    d_self = degree(self)
    if d_self < d_other:
      return self, False

    deg_diff = d_self-d_other
    a_d = self.coefs[d_self]
    mod = None if self.intmod != other.intmod else self.intmod
    mod_coef = lambda x,y: (x - a_d * y) % mod if mod != None else x - a_d * y
    coefs = [(mod_coef(c,other.coefs[k-deg_diff]) if  0 <= k-deg_diff < len(other.coefs) else c) for k,c in enumerate(self.coefs)]
    return Polynomial(coefs,mod), True

  def __mod__(self,other):
    p, t = self << other
    while t:
      p, t = p << other
    return p

  @staticmethod
  def random(intmod,dim,anchor = lambda v,w : random.randrange(w)):
    return Polynomial([anchor(i,intmod) for i in range(dim)],intmod)

  @staticmethod
  def randshift(coef,intmod,dim):
    degree_shift = [0]*(random.randrange(dim))
    return Polynomial(degree_shift + [coef % intmod],intmod)

  def __call__(self,arg=1):
    output = 0
    for i, c in enumerate(self.coefs):
      output = (output*arg+c) % self.intmod if self.intmod != None else output*arg+c
      if i == degree(self):
        return output
    return None

import math
from functools import reduce

def extended_gcd(a, b):
    r = [a,b]
    s = [1,0]
    t = [0,1]
    while not(0 in r):
      r1 = r[-1]
      r0 = r[-2]
      q, r2 = divmod(r0,r1)
      r.append(r2)
      s.append(s[-2]-q*s[-1])
      t.append(t[-2]-q*t[-1]) 
    return (r[-2],s[-2],t[-2])

def randinverse(intmod):
  a = random.randrange(1,intmod)
  while math.gcd(a,intmod) != 1:
    a = random.randrange(1,intmod)
  _, inva, k = extended_gcd(a,intmod)
  return (a,inva % intmod)

import numpy as np

class RandIso(object):
  def __init__(self,intmod,dim):
    self.intmod = intmod
    self.dim = dim
  def generate_pair(self):
    i = random.randrange(self.dim)
    j = random.choice([k for k in range(self.dim) if k!=i])
    return i,j
  def generate_swap(self):
    i,j = self.generate_pair()
    m = []
    for r in range(self.dim):
      if r == i:
        m.append([(1 if s==j else 0) for s in range(self.dim)])
      elif r == j:
        m.append([(1 if s==i else 0) for s in range(self.dim)])
      else:
        m.append([(1 if s==r else 0) for s in range(self.dim)])
    return np.array(m)
  def generate_mult(self):
    m = []
    invm = []
    for r in range(self.dim):
      a,inva = randinverse(self.intmod)
      m.append([(a if s==r else 0) for s in range(self.dim)])
      invm.append([(inva if s==r else 0) for s in range(self.dim)])
    return np.array(m), np.array(invm)
  def generate_line(self):
    i,j = self.generate_pair()
    m = []
    invm = []
    a = random.randrange(1,self.intmod)
    for r in range(self.dim):
      m.append([(1 if s==r else ( a if r==i and s==j else 0)) for s in range(self.dim)])
      invm.append([(1 if s==r else ( self.intmod-a if r==i and s==j else 0)) for s in range(self.dim)])
    return np.array(m), np.array(invm)
  def generate(self,length,pswap=1,pmult=2,pline=3):
    u = np.eye(self.dim, dtype=int)
    invu = np.eye(self.dim, dtype=int)
    choices = ["swap"] * pswap + ["mult"] * pmult + ["line"] * pline
    for i in range(length):
      x = random.choice(choices)
      if x == "swap":
        m = invm = self.generate_swap()
      if x == "mult":
        m , invm = self.generate_mult()
      if x == "line":
        m , invm = self.generate_line()
      u = (m @ u) % self.intmod
      invu = (invu @ invm) % self.intmod
    return (u % self.intmod), (invu % self.intmod)
class ArithChannel(object):
  #dim is both n = dim(x) and degre(u)
  def __init__(self,
                vanmod,
                intmod,
                dim,
                N,
                anchor = lambda v : 0 if random.uniform(0,1) < 0.5 else 1):
    # Implicit values for omega:
    self.omega = 1
    # Generate values for N, p, q, n, u, x, f0
    self.N = N
    if vanmod**2 < intmod and math.gcd(intmod,vanmod) == 1:
      self.vanmod = vanmod
      self.intmod = intmod
    else:
       self.vanmod = vanmod
       self.intmod = vanmod**2+1
    self.dim = dim
    self.q_list = factor_intmod(self.intmod)
    print(f"q_list={self.q_list}")
    self.u = self.generate_u()
    self.x, self.tensor = self.generate_secret(self.u)
    self.f0 = self.generate_initializer()
    self.f1, self.lvl_e,self.e = self.generate_noisy_key(anchor=anchor)
    self.n_repartition = self.generate_n_repartition()
    self.sigma_q_list = self.generate_sigma_q_list()

  def generate_vanisher(self, anchor = lambda i: random.randint(0,5)):
    e = []
    lvl_e = []
    for i in range(self.N):
      k = anchor(i)  
      randpoly = Polynomial.random(self.intmod,self.dim)

      diff_val = ((self.vanmod * k) % self.intmod - randpoly(arg=1)) % self.intmod

      shift = Polynomial.randshift(diff_val, self.intmod, self.dim)
      poly_e = (shift + randpoly).mod(self.intmod)
      e.append(poly_e)
      lvl_e.append(k) 
      # print(f"[C](e)={(shift + randpoly)(arg=1) % self.intmod}") # 実行した結果，問題なし
    return e, lvl_e
  def generate_initializer(self):
    """
    Generate initializer f0 such that each row of f0 is a multiple of (at least one) prime factor q_factor.
    """
    prime_factors = factor_intmod(self.intmod)
    if len(prime_factors)==0:
        raise ValueError("Failure to prime factorize: ", self.intmod)
    f0 = []
    for _ in range(self.N):
        row = []
        for _ in range(self.dim):
            q_factor = prime_factors[random.randrange(len(prime_factors))]
            randpoly = Polynomial.random(self.intmod, self.dim)
            multiplied = Polynomial(
                [(q_factor * c) % self.intmod for c in randpoly.coefs],
                self.intmod
            )
            # f0[i][j] = ( q_factor * randpoly ) mod u
            row.append( multiplied % self.u )
        f0.append(row)
    return f0
  def generate_noisy_key(self,anchor = lambda v : 0 if random.uniform(0,1) < 0.5 else 1):
    f1 = [Polynomial([0],self.intmod) for _ in range(self.N)]
    e, lvl_e = self.generate_vanisher(anchor = anchor)
    for i in range(self.N):
      for j in range(self.dim):
        f1[i] = (f1[i] + self.f0[i][j] * self.x[j]) % self.u
      f1[i] = f1[i] + e[i]
    return f1, lvl_e,e
    
  def publish(self,fhe = False):
    if fhe:
      return (self.f0,self.f1,self.vanmod,self.intmod,self.dim,self.N,self.u,self.tensor)
    else:
      return (self.f0,f1,self.vanmod,self.intmod,self.dim,self.N,self.u)

  def generate_u(self):
    #number of non zero coefficients for u where deg(u) = self.dim
    nonzeros = max(self.dim/2,min(self.dim-1,int(random.gauss(3*self.dim/4,self.dim/4))))
    
    #The dominant coefficient for u is equal to 1
    u_coefs = [1]

    while nonzeros-len(u_coefs) > 1:
      # a, _ = randinverse(self.intmod)
      a = random.randrange(self.intmod)
      u_coefs.append(a)

    u_coefs.append(self.intmod - sum(u_coefs))
    #number of zero coefficients for u
    zeros = self.dim - len(u_coefs)

    decomp = []
    remaining = zeros-sum(decomp)
    while remaining > 0:
      samp = max(0,min(remaining,int(random.gauss(remaining/2,remaining/2))))
      decomp.append(random.randint(0,samp))
      remaining = zeros-sum(decomp)
    u = []
    for i in range(len(u_coefs)):
      u.append(u_coefs[i])
      if i < len(decomp):
        u.extend([0] * decomp[i])
    u.extend([0] * (self.dim - len(u)+1))
    return Polynomial(u[::-1],self.intmod)
  
  def generate_n_repartition(self):
    return [random.randint(0,len(self.q_list)-1) for _ in range(self.dim)]

  def generate_sigma_q_list(self):
    ret = []
    for i in range(self.dim):
      row = []
      for j in range(self.dim):
        if i==j:
          row.append(self.intmod/self.q_list[self.n_repartition[j]])
        else:
          row.append(self.intmod/(self.q_list[self.n_repartition[i]] * self.q_list[self.n_repartition[j]]))
      ret.append(row)
    return ret
  def generate_secret(self,poly_u):
    ri = RandIso(self.intmod,self.dim)
    m, invm = ri.generate(60)
    x = []
    m_t = np.transpose(m)
    for k in range(len(m)):
      x.append(Polynomial(list(m_t[k]),self.intmod))
    
    tensor = []
    # we will have a list/array: tensor[i][j][k]
    # e_{i,j} \in \mathbb{Z}_q[X]_u,[C](e) = 0 mod q
    for i in range(len(x)):
      row = []
      for j in range(len(x)):
        xi_xj_mod_u = (x[i] * x[j]) % poly_u
        a_ij_poly = xi_xj_mod_u.mod(self.intmod)
        if sum(a_ij_poly.coefs[self.dim:]) != 0:
          print("Error in ArithChannel.generate_secret: tensor cannot be computed due to dimension discrepancies")
          print(a_ij_poly)
          exit()
        a_ij = np.array(a_ij_poly.coefs[:self.dim] + [0] * (self.dim - len(a_ij_poly.coefs)) )
        result =(invm @ a_ij) % self.intmod
        row.append(result)
      tensor.append(np.array(row))
    return x, np.array(tensor)
class ACESCipher(object):

  def __init__(self,dec,enc,lvl):
    self.dec = dec
    self.enc = enc
    self.uplvl = lvl
# Arithmetic Channel Ecnryption Scheme 
class ACES(object):

  def __init__(self,f0,f1,vanmod,intmod,dim,N,u):
    self.f0 = f0
    self.f1 = f1
    self.vanmod = vanmod
    self.intmod = intmod
    self.dim = dim
    self.N = N
    self.u = u

  def encrypt(self,m,anchor = lambda v,w: random.randint(0,w)):
    if m >= self.vanmod:
      print(f"Warning in ACES.encrypt: the input is equivalent to {m % self.vanmod}")
    b = self.generate_linear(anchor=anchor)
    enc = self.generate_error(m) # r(m)
    # print(f"[C](r(m))={enc(arg=1) % self.intmod},m={m}") # 実行した結果，問題なし
    for i in range(len(b)):
      enc = enc + (b[i] * self.f1[i]) % self.u # r(m) + b^T (f0+e) = r(m) + c
    dec = []
    for j in range(self.dim):
      dec_j = Polynomial([0],self.intmod)
      for i in range(self.N):
        dec_j = dec_j + (b[i] * self.f0[i][j]) % self.u # f_0^T b
      dec.append(dec_j)
    
    # return ACESCipher(dec,enc,self.N * self.vanmod) , [b[i](arg=1) for i in range(self.N)],b
    return ACESCipher(dec,enc,self.N * self.vanmod) , [b[i](arg=1) for i in range(self.N)]

  def generate_linear(self,anchor = lambda v,w: random.randint(0,w)):
    b = []
    for i in range(self.N):
      k = anchor(i,self.vanmod)
      randpoly = Polynomial.random(self.intmod,self.dim)
      shift = Polynomial.randshift(k - randpoly(arg=1),self.intmod,self.dim)
      # print(f"[C](bi)={(shift + randpoly)(arg=1)},[C](bi)<p:{(shift + randpoly)(arg=1) % self.intmod <= self.vanmod}") # 実行した結果，問題なし
      b.append((shift + randpoly)%self.u)
    return b

  def generate_error(self,m):
    randpoly = Polynomial.random(self.intmod,self.dim)
    shift = Polynomial.randshift(m - randpoly(arg=1),self.intmod,self.dim)
    return shift + randpoly

class ACESReader(object):

  def __init__(self,ac):
    self.x = ac.x
    self.vanmod = ac.vanmod
    self.intmod = ac.intmod
    self.dim = ac.dim
    self.N = ac.N
    self.u = ac.u

  def decrypt(self,c):
    cTx = Polynomial([0],self.intmod)
    sum = 0
    for i in range(self.dim):
      cTx += c.dec[i] * self.x[i] % self.u
      sum = (sum + c.dec[i](arg=1) * self.x[i](arg=1) ) % self.intmod
    m_pre = c.enc + Polynomial([-1],self.intmod) * cTx
    correct_m = ( m_pre(arg=1) % self.intmod ) % self.vanmod
    return correct_m
def test_print(str, calc_val, expected_val):
  if(calc_val == expected_val):
    # 緑で表示
    print(f"\033[32m{str}: {calc_val} == {expected_val} ? {calc_val == expected_val}\033[0m")
  else:
    # 赤で表示
    print(f"\033[31m{str}: {calc_val} == {expected_val} ? {calc_val == expected_val}\033[0m")

from pyaces.compaces import read_operations

class ACESAlgebra(object):

  def __init__(self,vanmod,intmod,dim,u,tensor):
    self.vanmod = vanmod
    self.intmod = intmod
    self.dim = dim
    self.u = u
    self.tensor = tensor

  def add(self,a,b):
    c0 = [ (a.dec[k]+b.dec[k]) % self.u for k in range(self.dim) ]
    c1 = (a.enc+b.enc) % self.u
    return ACESCipher(c0, c1, a.uplvl + b.uplvl)
  
  def mult(self,a,b):
    t = []
    for k in range(self.dim):
      tmp = Polynomial([0],self.intmod)
      for i in range(self.dim):
        for j in range(self.dim):
          tmp += Polynomial([self.tensor[i][j][k]],self.intmod) * a.dec[i] * b.dec[j]
      t.append(tmp)
    # t = c1 *\lambda c2
    c0 = [ ( b.enc * a.dec[k] +a.enc * b.dec[k] + Polynomial([-1],self.intmod) * t[k]) % self.u for k in range(self.dim) ] # c2'c1+c1'c2-c1 *\lambda c2
    c1 = ( a.enc * b.enc ) % self.u #c1'c2'
    # return ACESCipher(c0, c1, a.uplvl*b.uplvl*self.vanmod)
    return ACESCipher(c0, c1, (a.uplvl+b.uplvl+a.uplvl*b.uplvl)*self.vanmod)
  
  def compile(self,instruction):
    return lambda a: read_operations(self,instruction,a)

class ACESRefresher(object):
  def __init__(self,ac,algebra,encrypt,decrypt):
    self.vanmod = ac.vanmod
    self.intmod = ac.intmod
    self.dim = ac.dim
    self.N = ac.N
    self.u = ac.u
    self.ac = ac
    self.algebra = algebra
    self.encrypt = encrypt
    self.decrypt = decrypt

  def add(self,a,b):
    return a+b

  def mult(self,a,b):
    return a*b*self.vanmod

  def compile(self,instruction):
    return lambda a: read_operations(self,instruction,a)
  
  def generate_pseudocipertext(self,cipher:ACESCipher):
    #Here, the $n$-tuple $\intbrackets{\mathsf{C}}\tuplebrk{-c}$ is the tuple whose $i$-th coefficient is given by the following element in $\mathbb{Z}_q$.
    # \[
    # \intbrackets{\mathsf{C}}(-c_i) = -\intbrackets{\mathsf{C}}(c_i) = \pi_q(q - \iota_q(\intbrackets{\mathsf{C}}(c_i)))
    # \]
    C_negative_dec = [(self.intmod-(ci % self.u)(arg=1)) % self.intmod for ci in cipher.dec]
    C_enc = cipher.enc(arg=1)
    return C_negative_dec, C_enc  
  def generate_refresher(self,ac,secret):
    secret_q = []
    for i in range(self.dim):
      xi = Polynomial(secret[i].coefs,self.intmod)
      secret_q.append(xi)
    refresher_secret = [(xi % ac.u)(arg=1) % ac.vanmod for xi in secret_q]
    refresher_cipher_result = [self.encrypt(r) for r in refresher_secret]
    refresher_cipher,refresher_noise = zip(*refresher_cipher_result)
    return list(refresher_cipher),list(refresher_noise)

  def refresh(self,cipher,secret):
    # 1. Encrypt each component x_i of secret (= refresher)
    refresher_cipher, refresher_noise = self.generate_refresher(self.ac, secret)

    # 2. Extract pseudociphertext (c0, c1) of ciphertext cipher
    pse_c0, pse_c1 = self.generate_pseudocipertext(cipher) 

    # 3.integerize each component of c0+mod formatted
    pse_c0 = [p % self.vanmod for p in pse_c0] # \gamma1 [C](-ci)
    pse_c0_enc_tuple = [self.encrypt(pse_c0[i]) for i in range(len(pse_c0))]
    pse_c0_enc_cipher_tuple, pse_c0_enc_noise_tuple = [], []
    for txt, ns in pse_c0_enc_tuple:
        pse_c0_enc_cipher_tuple.append(txt)
        pse_c0_enc_noise_tuple.append(ns)
    # 4. c1 (scalar) encrypted
    pse_c1 = pse_c1 % self.vanmod # \gamma1 [C](c')
    pse_c1_enc_cipher, pse_c1_enc_noise = self.encrypt(pse_c1) 

    # 5. Calculates scalar product( c0_enc_tuple, refresher_cipher)
    scalar_product_result_cipher = self.scalar_product(pse_c0_enc_cipher_tuple,
                                                     refresher_cipher,)
    # 6. above + c1_enc_cipher to homomorphic add
    result = self.algebra.add(pse_c1_enc_cipher, scalar_product_result_cipher)

    # 7.
    # \kappa_0 &= \displaystyle p (k_2 +\sum_{i=1}^n (\kappa_i+k_{1,i} + \kappa_ik_{1,i}))
    kappa0 = self.vanmod * (
        sum(pse_c1_enc_noise)
        + sum(sum(noise) for noise in refresher_noise) 
        + sum(sum(ns) for ns in pse_c0_enc_noise_tuple)
        + sum(sum(rf_noise)*sum(c0_noise) for rf_noise,c0_noise in zip(refresher_noise,pse_c0_enc_noise_tuple))
    )
    # \kappa_1 &= \displaystyle \left\lfloor \frac{(p-1) + n(p-1)^2}{p} \right\rfloor
    kappa1 = int(((self.vanmod - 1) + self.dim*(self.vanmod - 1)**2) / self.vanmod)
    print(f"kappa0:{kappa0},kappa1:{kappa1}")

    new_uplvl = kappa0 + kappa1
    if new_uplvl < (self.intmod+1)/self.vanmod-1:
      print("\033[32mOK:")
    else:
      print("\033[31mNG:")
    print(f"new_uplvl:{new_uplvl},new_uplvl < (q+1)/p-1 : {new_uplvl} < {(self.intmod+1)/self.vanmod-1}\033[0m")
    return ACESCipher(result.dec, result.enc, new_uplvl)

  def scalar_product(self,a,b):
    current = self.algebra.mult(a[0],b[0])
    for i in range(1,len(a)):
      current = self.algebra.add(current,self.algebra.mult(a[i],b[i]))
    return current

  def gamma_1(self,zq):
    return zq(arg=1) % self.vanmod % self.intmod

  def is_refreshable(self,cipher,secret):
    ps_c0, ps_c1 = self.generate_pseudocipertext(cipher)
    ps_c0 = [p % self.intmod for p in ps_c0]
    ps_c1 = ps_c1 % self.intmod

    C_x = [(Polynomial(xi.coefs,self.intmod) % self.u)(arg=1) for xi in secret]
    # Compute [C]((-c))^T[C]((x))
    C_c_x_sum = 0
    for i in range(self.dim):
      C_c_x_sum += ps_c0[i] * C_x[i]
    rvalue = (ps_c1 + C_c_x_sum) % self.intmod
    lvalue = ps_c1 + C_c_x_sum
    lvalue_kpq = lvalue % (self.vanmod * self.intmod)
    # print(f"lvalue:{lvalue},lvalue_kpq:{lvalue_kpq},rvalue:{rvalue}")
    rvalue2 = rvalue % self.vanmod
    lvalue2 = ps_c1 % self.vanmod + C_c_x_sum % self.vanmod
    # print(f"lvalue2:{lvalue2},rvalue2:{rvalue2}")
    return lvalue2 == rvalue2

def factor_intmod(q):
    factors = []
    x = q
    d = 2
    while d*d <= x:
        while x % d == 0:
            factors.append(d)
            x //= d
        d += 1 if d==2 else 2
    if x>1:
        factors.append(x)
    return factors

