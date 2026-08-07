"""Exact quadratization of the full-tensor portfolio HUBO."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, Iterable, Mapping, Tuple
import numpy as np
from .higher_moment_extension import SourceMomentData, bit_matrix
from .source_hubo_models import SourceHuboConfig, source_hubo_state_table
Monomial=Tuple[int,...]; Polynomial=Dict[Monomial,float]
class AncillaBudgetError(ValueError): pass
@dataclass(frozen=True)
class ProductConstraint:
    ancilla:int; left:int; right:int
@dataclass(frozen=True)
class QuadratizationResult:
    original_polynomial:Mapping[Monomial,float]; reduced_objective:Mapping[Monomial,float]; qubo_polynomial:Mapping[Monomial,float]; constraints:Tuple[ProductConstraint,...]; n_original_variables:int; n_ancillas:int; n_total_variables:int; penalty_strength:float; objective_range_bound:float; Q:np.ndarray; offset:float
def _monomial(variables:Iterable[int])->Monomial: return tuple(sorted(set(int(v) for v in variables)))
def _add(poly:Polynomial,variables:Iterable[int],coefficient:float,*,tol:float=1e-15)->None:
    value=float(coefficient)
    if abs(value)<=tol:return
    key=_monomial(variables); new=poly.get(key,0.0)+value
    if abs(new)<=tol: poly.pop(key,None)
    else: poly[key]=float(new)
def _bit_variable(asset:int,bit:int,bits_per_asset:int)->int:return int(asset*bits_per_asset+bit)
def portfolio_hubo_polynomial(data:SourceMomentData,config:SourceHuboConfig,*,prune_tolerance:float=1e-14)->Polynomial:
    n_assets=len(data.tickers); bpa=int(config.bits_per_asset); powers=(2**np.arange(bpa)).astype(float); poly={}
    for i in range(n_assets):
        for bi in range(bpa): _add(poly,(_bit_variable(i,bi,bpa),),-config.lambda_return*data.expected_returns[i]*powers[bi])
    for i in range(n_assets):
        for j in range(n_assets):
            base=config.lambda_variance*data.covariance[i,j]
            if abs(base)<=prune_tolerance:continue
            for bi in range(bpa):
                vi=_bit_variable(i,bi,bpa)
                for bj in range(bpa): _add(poly,(vi,_bit_variable(j,bj,bpa)),base*powers[bi]*powers[bj])
    for i in range(n_assets):
        for j in range(n_assets):
            for k in range(n_assets):
                base=-config.lambda_skewness*data.co_skewness[i,j,k]
                if abs(base)<=prune_tolerance:continue
                for bi in range(bpa):
                    vi=_bit_variable(i,bi,bpa)
                    for bj in range(bpa):
                        vj=_bit_variable(j,bj,bpa)
                        for bk in range(bpa): _add(poly,(vi,vj,_bit_variable(k,bk,bpa)),base*powers[bi]*powers[bj]*powers[bk])
    for i in range(n_assets):
        for j in range(n_assets):
            for k in range(n_assets):
                for l in range(n_assets):
                    base=config.lambda_kurtosis*data.co_kurtosis[i,j,k,l]
                    if abs(base)<=prune_tolerance:continue
                    for bi in range(bpa):
                        vi=_bit_variable(i,bi,bpa)
                        for bj in range(bpa):
                            vj=_bit_variable(j,bj,bpa)
                            for bk in range(bpa):
                                vk=_bit_variable(k,bk,bpa)
                                for bl in range(bpa): _add(poly,(vi,vj,vk,_bit_variable(l,bl,bpa)),base*powers[bi]*powers[bj]*powers[bk]*powers[bl])
    linear=[(_bit_variable(i,bi,bpa),powers[bi]) for i in range(n_assets) for bi in range(bpa)]; target=float(config.target_units); _add(poly,(),config.lambda_budget*target*target)
    for variable,weight in linear:_add(poly,(variable,),-2.0*config.lambda_budget*target*weight)
    for va,wa in linear:
        for vb,wb in linear:_add(poly,(va,vb),config.lambda_budget*wa*wb)
    return {key:value for key,value in poly.items() if abs(value)>prune_tolerance}
def polynomial_degree(poly):return max((len(key) for key in poly),default=0)
def polynomial_range_bound(poly):return float(sum(abs(value) for key,value in poly.items() if key))
def _pair_scores(poly):
    scores={}
    for key,coefficient in poly.items():
        degree=len(key)
        if degree<=2:continue
        weight=abs(float(coefficient))*float(degree-2)
        for pair in combinations(key,2):scores[pair]=scores.get(pair,0.0)+weight
    return scores
def _replace_pair(poly,pair,ancilla):
    a,b=pair; result={}
    for key,coefficient in poly.items():
        variables=set(key)
        if len(key)>2 and a in variables and b in variables: variables.remove(a); variables.remove(b); variables.add(int(ancilla)); _add(result,variables,coefficient)
        else:_add(result,key,coefficient)
    return result
def _add_rosenberg_penalty(poly,constraint,penalty):
    a,b,y=constraint.left,constraint.right,constraint.ancilla; _add(poly,(a,b),penalty); _add(poly,(a,y),-2*penalty); _add(poly,(b,y),-2*penalty); _add(poly,(y,),3*penalty)
def polynomial_to_qubo(poly,n_variables):
    if polynomial_degree(poly)>2:raise ValueError("Polynomial is not quadratic.")
    Q=np.zeros((int(n_variables),int(n_variables)),float); offset=float(poly.get((),0.0))
    for key,coefficient in poly.items():
        if not key:continue
        if len(key)==1:Q[key[0],key[0]]+=float(coefficient)
        else:
            i,j=key; half=.5*float(coefficient); Q[i,j]+=half; Q[j,i]+=half
    return Q,offset
def exact_quadratize_hubo(polynomial,n_original_variables,*,max_ancillas=None,penalty_strength=None,penalty_safety_factor=1.05):
    if n_original_variables<=0:raise ValueError("n_original_variables must be positive.")
    if max_ancillas is not None and max_ancillas<0:raise ValueError("max_ancillas must be nonnegative or None.")
    if penalty_safety_factor<=1.0 and penalty_strength is None:raise ValueError("Automatic exact penalty requires penalty_safety_factor > 1.")
    original={tuple(key):float(value) for key,value in polynomial.items()}; reduced=dict(original); constraints=[]; pair_to_ancilla={}; next_variable=int(n_original_variables)
    while polynomial_degree(reduced)>2:
        scores=_pair_scores(reduced)
        if not scores:raise RuntimeError("High-order polynomial remains but no reducible pair was found.")
        pair=sorted(scores,key=lambda pair:(pair not in pair_to_ancilla,-scores[pair],pair))[0]; ancilla=pair_to_ancilla.get(pair)
        if ancilla is None:
            if max_ancillas is not None and len(constraints)>=max_ancillas:raise AncillaBudgetError(f"Exact quadratization needs more than {max_ancillas} ancillas; residual polynomial degree is {polynomial_degree(reduced)}.")
            ancilla=next_variable; next_variable+=1; pair_to_ancilla[pair]=ancilla; constraints.append(ProductConstraint(ancilla,pair[0],pair[1]))
        reduced=_replace_pair(reduced,pair,ancilla)
    range_bound=polynomial_range_bound(reduced)
    if penalty_strength is None: penalty=float(penalty_safety_factor*range_bound+1e-12)
    else:
        penalty=float(penalty_strength)
        if penalty<=range_bound:raise ValueError(f"For certified exactness, penalty_strength must be strictly larger than the reduced-objective range bound {range_bound:.12g}.")
    qubo_poly=dict(reduced)
    for constraint in constraints:_add_rosenberg_penalty(qubo_poly,constraint,penalty)
    n_total=int(n_original_variables+len(constraints)); Q,offset=polynomial_to_qubo(qubo_poly,n_total)
    return QuadratizationResult(original,reduced,qubo_poly,tuple(constraints),int(n_original_variables),len(constraints),n_total,penalty,range_bound,Q,offset)
def quadratize_portfolio_hubo(data,config,*,max_ancillas=None,penalty_strength=None,penalty_safety_factor=1.05):
    poly=portfolio_hubo_polynomial(data,config); return exact_quadratize_hubo(poly,len(data.tickers)*config.bits_per_asset,max_ancillas=max_ancillas,penalty_strength=penalty_strength,penalty_safety_factor=penalty_safety_factor)
def lift_original_bits(bits,result):
    values=np.asarray(bits,dtype=np.int8); single=values.ndim==1; matrix=values[None,:] if single else values
    if matrix.shape[1]!=result.n_original_variables:raise ValueError("Original bit width does not match quadratization.")
    lifted=np.zeros((matrix.shape[0],result.n_total_variables),dtype=np.int8); lifted[:,:result.n_original_variables]=matrix
    for c in result.constraints:lifted[:,c.ancilla]=lifted[:,c.left]*lifted[:,c.right]
    return lifted[0] if single else lifted
def constraints_satisfied(bits,result):
    values=np.asarray(bits,dtype=np.int8); single=values.ndim==1; matrix=values[None,:] if single else values; ok=np.ones(matrix.shape[0],bool)
    for c in result.constraints:ok&=matrix[:,c.ancilla]==matrix[:,c.left]*matrix[:,c.right]
    return bool(ok[0]) if single else ok
def evaluate_polynomial(poly,bits):
    values=np.asarray(bits,float); single=values.ndim==1; matrix=values[None,:] if single else values; energy=np.full(matrix.shape[0],float(poly.get((),0.0)),float)
    for key,coefficient in poly.items():
        if key:energy+=float(coefficient)*np.prod(matrix[:,key],axis=1)
    return float(energy[0]) if single else energy
def evaluate_qubo(result,bits):
    values=np.asarray(bits,float); single=values.ndim==1; matrix=values[None,:] if single else values; energy=np.einsum("ni,ij,nj->n",matrix,result.Q,matrix,optimize=True)+result.offset; return float(energy[0]) if single else energy
def compare_qubo_with_hubo(data,config,result=None,*,max_full_enumeration_variables=22,atol=1e-8):
    quad=quadratize_portfolio_hubo(data,config) if result is None else result; table=source_hubo_state_table(data,config); original_bits=np.asarray(table["bits"],np.int8); source_energy=np.asarray(table["total_energy"],float); polynomial_energy=np.asarray(evaluate_polynomial(quad.original_polynomial,original_bits),float); lifted=lift_original_bits(original_bits,quad); lifted_qubo_energy=np.asarray(evaluate_qubo(quad,lifted),float); source_poly_error=float(np.max(np.abs(source_energy-polynomial_energy))); hubo_qubo_error=float(np.max(np.abs(source_energy-lifted_qubo_energy))); hubo_ground=int(np.argmin(source_energy)); lifted_ground=int(np.argmin(lifted_qubo_energy))
    audit={"n_original_variables":quad.n_original_variables,"n_ancillas":quad.n_ancillas,"n_total_variables":quad.n_total_variables,"original_hubo_degree":polynomial_degree(quad.original_polynomial),"reduced_objective_degree":polynomial_degree(quad.reduced_objective),"qubo_degree":polynomial_degree(quad.qubo_polynomial),"penalty_strength":quad.penalty_strength,"objective_range_bound":quad.objective_range_bound,"source_vs_polynomial_max_abs_error":source_poly_error,"hubo_vs_lifted_qubo_max_abs_error":hubo_qubo_error,"lifted_constraints_all_satisfied":bool(np.all(constraints_satisfied(lifted,quad))),"hubo_ground_state_index":hubo_ground,"lifted_qubo_ground_state_index":lifted_ground,"lifted_ground_matches_hubo":bool(lifted_ground==hubo_ground),"energy_equivalence_passed":bool(source_poly_error<=atol and hubo_qubo_error<=atol),"full_qubo_enumeration_performed":False}
    if quad.n_total_variables<=int(max_full_enumeration_variables):
        all_qubo_bits=bit_matrix(quad.n_total_variables); all_qubo_energy=np.asarray(evaluate_qubo(quad,all_qubo_bits),float); q_index=int(np.argmin(all_qubo_energy)); q_bits=all_qubo_bits[q_index]; consistent=bool(constraints_satisfied(q_bits,quad)); projected=q_bits[:quad.n_original_variables]; projected_index=int(sum(int(projected[i])<<i for i in range(quad.n_original_variables))); audit.update({"full_qubo_enumeration_performed":True,"full_qubo_states_enumerated":int(2**quad.n_total_variables),"full_qubo_ground_constraint_consistent":consistent,"full_qubo_projected_original_state_index":projected_index,"full_qubo_ground_matches_hubo":bool(consistent and projected_index==hubo_ground)})
    else:audit["full_qubo_enumeration_skip_reason"]=f"{quad.n_total_variables} variables exceed the configured exact-enumeration limit {max_full_enumeration_variables}. Lifted-subspace equivalence was still audited exactly."
    return audit
def export_qubo_upper_triangle(result):
    terms={}; n=result.n_total_variables
    for i in range(n):
        diagonal=float(result.Q[i,i])
        if abs(diagonal)>1e-15:terms[(i,i)]=diagonal
        for j in range(i+1,n):
            coefficient=float(result.Q[i,j]+result.Q[j,i])
            if abs(coefficient)>1e-15:terms[(i,j)]=coefficient
    return terms
