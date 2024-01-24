Production of Input-Output matrix, given a list of Brazilian municipalities. Following work done by 

Pangallo et al. 2023, which follows Miller, R. E. & Blair, P. D. Input–Output Analysis: Foundations and Extensions (Cambridge Univ. Press, 2009); 
https://doi.org/10.1017/CBO9780511626982 and 

Bonfiglio, A. & Chelli, F. Assessing the behaviour of non-survey methods for constructing regional input–output tables 
through a Monte Carlo simulation. Economic Systems Research 20, 243–258 (2008)

# From the literature
1. The aim is to ''split the intermediate consumption matriz into matrices of the value of intermediate goods flowing from industries in METRO to METRO, from METRO to REST, from REST to METRO, and from REST to REST.''
2. That division would also apply to final demand table
3. We have to gather inputs from the REST when not produced in METRO. We also may sell output to other metro regions (REST)
4. The end-goal is to have rho_METRO_kl to indicate "how much intra-regional trade flows DIFFER from national ones (our emphasis)", which is usually < 1

# Goal
Gather sensible estimates of different regions input-output tables of technical coefficients and final demand
# Steps
1. Calculate **Simple Location Quotient** (SLQ)
2. Calculate **Cross-Industry Location Quotient** (CILQ)
3. Calculate FLQ_kl
4. Calculate ro^r_kl
5. 