import numpy as np
from openmm import openmm, unit

def genParamsDH(temp,ionic):
    """ Debye-Huckel parameters. """

    kT = 8.3145*temp*1e-3
    # Calculate the prefactor for the Yukawa potential
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.02214076*1000/kT
    eps_yu = lB*kT
    # Calculate the inverse of the Debye length
    k_yu = np.sqrt(8*np.pi*lB*ionic*6.02214076/10)
    return eps_yu, k_yu

def init_bonded_interactions():
    """ Define bonded interactions. """

    # harmonic bonds
    hb = openmm.HarmonicBondForce()
    hb.setUsesPeriodicBoundaryConditions(True)

    return hb

def init_ah_interactions(eps,rc,fixed_lambda):
    """ Define Ashbaugh-Hatch interactions. """

    # intermolecular interactions
    energy_expression = f'{eps}*select(step(r-2^(1/6)*s),4*l*((s/r)^12-(s/r)^6-shift),4*((s/r)^12-(s/r)^6-l*shift)+(1-l))'
    #ah = openmm.CustomNonbondedForce(energy_expression+f'; s=0.5*(s1+s2); l=0.5*(l1+l2); shift=(0.5*(s1+s2)/{rc})^12-(0.5*(s1+s2)/{rc})^6')
    ah = openmm.CustomNonbondedForce(energy_expression+f'; l=select(id1+id2,(id1*id2)*0.5*(l1+l2),{fixed_lambda}); shift=(s/{rc})^12-(s/{rc})^6; s=0.5*(s1+s2)')

    ah.addPerParticleParameter('s')
    ah.addPerParticleParameter('l')
    ah.addPerParticleParameter('id')

    ah.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    ah.setCutoffDistance(rc*unit.nanometer)
    ah.setForceGroup(0)

    print('Ashbaugh-Hatch potential between particles with lambda=1 and sigma=0.68 at',rc*unit.nanometer,end=': ')
    print(4*eps*((0.68/rc)**12-(0.68/rc)**6)*unit.kilojoules_per_mole)
    return ah

def init_yu_interactions(eps, k, rc):
    """ Define Yukawa interactions. """

    shift = np.exp(-k*rc)/rc
    yu = openmm.CustomNonbondedForce(f'q*{eps}*(exp(-{k}*r)/r-{shift}); q=q1*q2')
    yu.addPerParticleParameter('q')

    print('Debye-Hückel potential between unit charges at',rc*unit.nanometer,end=': ')
    print(eps*shift*unit.kilojoules_per_mole)

    yu.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    yu.setCutoffDistance(rc*unit.nanometer)
    yu.setForceGroup(1)

    return yu

def init_nonbonded_interactions(eps_lj,cutoff_lj,eps_yu,k_yu,cutoff_yu,fixed_lambda):
    """ Define protein interaction expressions (without restraints). """

    ah = init_ah_interactions(eps_lj, cutoff_lj, fixed_lambda)
    yu = init_yu_interactions(eps_yu, k_yu, cutoff_yu)

    return ah, yu

def init_angles():
    ha = openmm.HarmonicAngleForce()
    ha.setUsesPeriodicBoundaryConditions(True)
    return ha

def init_lipid_interactions(eps_lj, eps_yu, cutoff_yu, factor=1.9):
    """ Define lipid interaction expressions. """

    # harmonic angles
    cos = init_cosine_interactions(factor*eps_lj)
    cn = init_charge_nonpolar_interactions(eps_yu, cutoff_yu)
    return cos, cn

def init_wcafene(eps_lj):
    wcafene = init_wcafene_interactions(3*eps_lj)
    return wcafene

def init_restraints(restraint_type):
    """ Initialize restraints. """

    if restraint_type == 'harmonic':
        cs = openmm.HarmonicBondForce()
    if restraint_type == 'go':
        go_expr = 'k*(5*(s/r)^12-6*(s/r)^10)'
        cs = openmm.CustomBondForce(go_expr+'; s=s; k=k')#; shift=(0.5*(s)/rc)^12-(0.5*(s)/rc)^6')
        cs.addPerBondParameter('s')
        cs.addPerBondParameter('k')
    cs.setUsesPeriodicBoundaryConditions(True)
    return cs

def init_scaled_LJ(eps_lj,cutoff_lj):
    """ Initialize restraints. """

    energy_expression = 'select(step(r-2^(1/6)*s),n*4*eps*l*((s/r)^12-(s/r)^6-shift),n*4*eps*((s/r)^12-(s/r)^6-l*shift)+n*eps*(1-l))'
    scLJ = openmm.CustomBondForce(energy_expression+'; shift=(s/rc)^12-(s/rc)^6')
    scLJ.addGlobalParameter('eps',eps_lj*unit.kilojoules_per_mole)
    scLJ.addGlobalParameter('rc',float(cutoff_lj)*unit.nanometer)
    scLJ.addPerBondParameter('s')
    scLJ.addPerBondParameter('l')
    scLJ.addPerBondParameter('n')
    scLJ.setUsesPeriodicBoundaryConditions(True)
    return scLJ

def init_scaled_YU(eps_yu,k_yu):
    """ Initialize restraints. """

    shift = np.exp(-k_yu*4.0)/4.0
    scYU = openmm.CustomBondForce(f'n*q*{eps_yu}*(exp(-{k_yu}*r)/r-{shift})')
    scYU.addPerBondParameter('q')
    scYU.addPerBondParameter('n')
    scYU.setUsesPeriodicBoundaryConditions(True)
    return scYU

def init_slab_restraints(box,k):
    """ Define restraints towards box center in z direction. """

    mindim = np.amin(box)
    rcent_expr = 'k*abs(periodicdistance(x,y,z,x,y,z0))'
    rcent = openmm.CustomExternalForce(rcent_expr)
    rcent.addGlobalParameter('k',k*unit.kilojoules_per_mole/unit.nanometer)
    rcent.addGlobalParameter('z0',box[2]/2.*unit.nanometer) # center of box in z
    # rcent.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    # rcent.setCutoffDistance(mindim/2.*unit.nanometer)
    return rcent

def add_single_restraint(
        cs, restraint_type: str,
        dij: float, k: float,
        i: int, j: int):
    """ Add single harmonic or Go restraint. """

    if restraint_type == 'harmonic':
        cs.addBond(
                i,j, dij*unit.nanometer,
                k*unit.kilojoules_per_mole/(unit.nanometer**2))
    elif restraint_type == 'go':
        cs.addBond(
                i,j, [dij*unit.nanometer,
                k*unit.kilojoules_per_mole])
    restr_pair = [i+1, j+1, dij, k] # 1-based
    return cs, restr_pair

def add_scaled_lj(scLJ, i, j, offset, comp):
    """ Add downscaled LJ interaction. """

    s = 0.5 * (comp.sigmas[i] + comp.sigmas[j])
    l = 0.5 * (comp.lambdas[i] + comp.lambdas[j])
    scLJ.addBond(i+offset,j+offset, [s*unit.nanometer, l*unit.dimensionless, comp.bondscale[i,j]*unit.dimensionless])
    scaled_pair = [i+offset+1, j+offset+1, s, l, comp.bondscale[i,j]] # 1-based
    return scLJ, scaled_pair

def add_scaled_yu(scYU, i, j, offset, comp):
    """ Add downsscaled YU interaction. """

    qij = comp.qs[i] * comp.qs[j] * unit.dimensionless
    scYU.addBond(i+offset, j+offset, [qij, comp.bondscale[i,j]*unit.dimensionless])
    scaled_pair = [i+offset+1, j+offset+1, comp.bondscale[i,j]] # 1-based
    return scYU, scaled_pair

def add_exclusion(force, i: int, j: int):
    """ Add exclusions to a list of openMM forces """
    force.addExclusion(i,j)
    return force

def init_wcafene_interactions(eps):
    """ Define FENE interaction. """

    wca_expression = f'4*{eps}*select(step(r-2^(1/6)*s),0,(s/r)^12-(s/r)^6+1/4)'
    fene_expression = '+ -0.5*kfene*(rinf^2)*log(1-(r/rinf)^2); rinf=1.5*s'
    wcafene = openmm.CustomBondForce(wca_expression+fene_expression)
    wcafene.addPerBondParameter('s')
    wcafene.addPerBondParameter('kfene')
    wcafene.setUsesPeriodicBoundaryConditions(True)
    return wcafene

def init_cosine_interactions(eps):
    """ Define cosine interaction (Cooke and Deserno lipid model, DOI: https://doi.org/10.1063/1.2135785). """

    cosine_expression = f'prefactor*select(step(r-rc-1.5*s),0,select(step(r-rc),-{eps}*(cos({np.pi}*(r-rc)/(2*1.5*s)))^2,-{eps}))'
    cosine = openmm.CustomNonbondedForce(cosine_expression+'; prefactor=select(id1*id2,1-delta(l1*l2),(id1+id2)*l1*l2); rc=2^(1/6)*s; s=0.5*(s1+s2)')
    cosine.addPerParticleParameter('s')
    cosine.addPerParticleParameter('l')
    cosine.addPerParticleParameter('id')
    cosine.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    cosine.setCutoffDistance((2**(1/6)+1.5)*unit.nanometer)
    cosine.setForceGroup(2)
    return cosine

def init_charge_nonpolar_interactions(eps,rc):
    """ Define charge-nonpolar interaction (lipid model, DOI: https://doi.org/10.1063/1.5058234 and DOI: https://doi.org/10.1073/pnas.2311700120). """

    cn = openmm.CustomNonbondedForce(f'-step(id1+id2)*{eps}*alphaq2R3/2*(1/r-1/{rc}); alphaq2R3=alpha1*q2^2*R31+alpha2*q1^2*R32')
    cn.addPerParticleParameter('R3')
    cn.addPerParticleParameter('alpha')
    cn.addPerParticleParameter('q')
    cn.addPerParticleParameter('id')
    cn.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    cn.setCutoffDistance(rc*unit.nanometer)
    cn.setForceGroup(1)
    return cn

def genMegoParams(md_mat, rc_mat, epsilon_0, sigmas, stickyness, epsilon_calvados):
    """ Generate parameters for the multi-eGO potential. 
    Creates the force field as in https://github.com/frantropy/multi-eGO/blob/419f1ad7418eb7099bf2231b9459e7271ddfc57d/src/multiego/ensemble.py#L904 """

    print('Generating multi-eGO parameters...')

    md_threshold = 0.05 # TODO implement
    epsilon_min = 0.07

    combined_sigmas = np.sqrt(sigmas * sigmas[:, np.newaxis]).flatten()

    r_min = 2 ** ( 1 / 6 ) * combined_sigmas
    combined_lambdas = 0.5 * (stickyness + stickyness[:, np.newaxis]).flatten()
    lj_min = 4 * epsilon_calvados * ( ( combined_sigmas / r_min ) ** 12 - ( combined_sigmas / r_min ) ** 6 ) \
             - combined_lambdas * ( 4 * epsilon_calvados * ( ( combined_sigmas / r_min ) ** 12 - ( combined_sigmas / r_min ) ** 6 ) ) \
             - epsilon_calvados * ( 1 - combined_lambdas )
    epsilon_prior = lj_min - combined_lambdas * lj_min - epsilon_calvados * ( 1 - combined_lambdas )

    rc_threshold = md_threshold ** ( (epsilon_0 - epsilon_prior) / (epsilon_0 - epsilon_min) )
    limit_rc_att = rc_threshold ** ( (epsilon_prior - epsilon_min) / (epsilon_0 - epsilon_prior) )

    pdmat = md_mat[:, 3]
    pdmat_rc = rc_mat[:, 3]
    distance = md_mat[:, 4]
    rc_distance = rc_mat[:, 4]

    rep = 4 * epsilon_prior * combined_sigmas ** 12 # should be c12s

    mask = pdmat <= md_threshold
    distance = np.where(
        epsilon_prior < 0,
        combined_sigmas * 2.0 ** (1.0 / 6.0) / (epsilon_0 ** (1.0 / 12.0)),
        combined_sigmas * 2.0 ** (1.0 / 6.0)
    )

    mask_2 = pdmat_rc <= md_threshold
    rc_distance = np.where(
        epsilon_prior < 0,
        combined_sigmas * 2.0 ** (1.0 / 6.0) / (epsilon_0 ** (1.0 / 12.0)),
        combined_sigmas * 2.0 ** (1.0 / 6.0)
    )

    # is this ever the case?
    epsilon = np.where(epsilon_prior < 0, - rep * (distance / rc_distance) ** 12, 0.0)

    # Attractive interactions
    epsilon = np.where(
        (pdmat > limit_rc_att * np.maximum(pdmat_rc, rc_threshold)) & (pdmat > md_threshold),
        np.maximum(0.0, epsilon_prior) - (epsilon_0 - np.maximum(0.0, epsilon_prior)) / np.log(rc_threshold) * np.log(pdmat / (np.maximum(pdmat_rc, rc_threshold))),
        epsilon
    )

    epsilon = np.where(
        (pdmat > limit_rc_att * np.maximum(pdmat_rc, rc_threshold)) & (pdmat < md_threshold),
        np.maximum(0.0, (distance ** 12) * ((epsilon_0 - np.maximum(0.0, epsilon_prior)) / (np.log(rc_threshold))) * np.log(pdmat / (np.maximum(pdmat_rc, rc_threshold)))) - rep * (distance / rc_distance) ** 12,
        epsilon
    )
    # clean NaNs
    # meGO_LJ.dropna(subset=["epsilon"], inplace=True)
    # meGO_LJ = meGO_LJ[meGO_LJ.epsilon != 0]

    # lower value for repulsion
    epsilon = np.where((epsilon < 0.0) & (-epsilon < 0.02 * rep), -0.02 * rep, epsilon)
    # higher value for repulsion
    epsilon = np.where((epsilon < 0.0) & (-epsilon > 2.0 * rep), -2.0 * rep, epsilon)
    # but within a lower
    epsilon = np.where((epsilon < 0.0) & (-epsilon < 0.2 * rep), -0.2 * rep, epsilon)
    # and an upper value
    epsilon = np.where((epsilon < 0.0) & (-epsilon > 2.0 * rep), -2.0 * rep, epsilon)

    mego_sigma = distance / 2 ** (1.0 / 6.0)
    # sigma boundaries for attractive interactions
    mego_sigma = np.where(
        (epsilon > 0.0) & (mego_sigma < 0.7 * combined_sigmas),
        0.7 * combined_sigmas,
        mego_sigma
    )
    mego_sigma = np.where(
        (epsilon > 0.0) & (mego_sigma > 1.3 * combined_sigmas),
        1.3 * combined_sigmas,
        mego_sigma
    )
    mego_sigma = np.where(epsilon < 0.0, - epsilon ** (1.0 / 12.0), mego_sigma)

    # Calculate c6 and c12 for 
    c6 = np.where(epsilon < 0.0, 0.0, 4 * epsilon * (mego_sigma ** 6))
    c12 = np.where(epsilon < 0.0, -epsilon, 4 * epsilon * (mego_sigma ** 12))

    return c6, c12, mego_sigma

def init_mego_interactions(mego_c6, mego_c12, fixed_lambda, s, rc):
    """ Define a multi-eGO interaction. """
    mego_c6 = mego_c6.reshape((int(np.sqrt(len(mego_c6))), int(np.sqrt(len(mego_c6)))))
    mego_c12 = mego_c12.reshape((int(np.sqrt(len(mego_c12))), int(np.sqrt(len(mego_c12)))))

    energy_expressions = []
    energy_expression = lambda c6, c12, s, pid1, pid2: f'select(step(abs(id1-{pid1})+abs(id2-{pid2}),0,select(step(r-2^(1/6)*{s}),l*(({c12}/r^12-{c6}/r^6)-shift),(({c12}/r^12-{c6}/r^6)-l*shift)+(1-l));shift=({s}/{rc})^12-({s}/{rc})^6; l=select(id1+id2,(id1*id2)*0.5*(l1+l2),{fixed_lambda}))'

    energy_expression = lambda c6, c12, s, pid1, pid2: f'select(step(abs(id1-{pid1})+abs(id2-{pid2})),select(step(r-2^(1/6)*{s}),l*(({c12}/r^12-{c6}/r^6)-shift),(({c12}/r^12-{c6}/r^6)-l*shift)+(1-l));shift=({s}/{rc})^12-({s}/{rc})^6; l=select(id1+id2,(id1*id2)*0.5*(l1+l2),{fixed_lambda})),0)'

    energy_expression = lambda c6, c12, s, pid1, pid2: (
        f'select('
            f'step('
                f'abs(id1-{pid1})+abs(id2-{pid2})'
            f'),'
            f'select('
                f'step(r-2^(1/6)*{s}),'
                f'l*(({c12}/r^12-{c6}/r^6)-shift),'
                f'(({c12}/r^12-{c6}/r^6)-l*shift)+(1-l)'
            f'),'
            f'0'
        f')'
        f';shift=({s}/{rc})^12-({s}/{rc})^6;'
        f'l=select('
            f'id1+id2,'
            f'(id1*id2)*0.5*(l1+l2),'
            f'{fixed_lambda}'
        f')'
    )

    for ai in range(len(mego_c6)):
        for aj in range(len(mego_c6)):
            sig = 0.5 * (s[ai] + s[aj])
            print(mego_c6)
            c6 = mego_c6[ai][aj]
            c12 = mego_c12[ai][aj]
            mego = openmm.CustomNonbondedForce(energy_expression(c6, c12, sig, ai, aj))
            mego.addPerParticleParameter('l')
            mego.addPerParticleParameter('id')
            mego.addPerParticleParameter('pid')
            mego.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
            mego.setCutoffDistance(rc*unit.nanometer)
            mego.setForceGroup(0)

            energy_expressions.append(mego)

    return energy_expressions
