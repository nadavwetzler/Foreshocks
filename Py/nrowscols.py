from math import gcd
import numpy as np
import math

def is_prime(n):
  if n == 2 or n == 3: return True
  if n < 2 or n%2 == 0: return False
  if n < 9: return True
  if n%3 == 0: return False
  r = int(n**0.5)
  # since all primes > 3 are of the form 6n ± 1
  # start with f=5 (which is prime)
  # and test f, f+2 for being prime
  # then loop by 6. 
  f = 5
  while f <= r:
    print('\t',f)
    if n % f == 0: return False
    if n % (f+2) == 0: return False
    f += 6
  return True    

def get_factor_list(n):

    """
    Use trial division to identify the factors of n.
    1 is always a factor of any integer so is added at the start.
    We only need to check up to n/2, and then add n after the loop.
    """

    factors = [1]

    for t in range(2, (math.ceil((n / 2) + 1))):
        if n % t == 0:
            factors.append(t)

    factors.append(n)

    return factors

def factorization(n):
    factors = []
    def get_factor(n):
        x_fixed = 2
        cycle_size = 2
        x = 2
        factor = 1

        while factor == 1:
            for count in range(cycle_size):
                if factor > 1: break
                x = (x * x + 1) % n
                factor = gcd(x - x_fixed, n)

            cycle_size *= 2
            x_fixed = x
        return factor

    while n > 1:
        next = get_factor(n)
        factors.append(next)
        n //= next

    return factors


def numSubplots(n):
    while is_prime(n) & n>4:
        n = n + 1

    # p = factorization(n)
    p = get_factor_list(n)

    if len(p) == 1:
        p = [1, p[0]]

    else:
        while len(p) > 2:
            if len(p) >= 4:
                p[0] = p[0]*p[-1]
                p[1] = p[1]*p[-1]
            else:
                p[0] = p[0]*p[1]
                p[1] = []
            p = np.sort(p)
    p = p[0:2]
    return p, n

def colsrows(n):
    p,n = numSubplots(n)
    while p[1] / p[0] > 2.5:
        N=n+1
        p,n = numSubplots(N)
    return p


p,n = colsrows(22)
print(p, n)

