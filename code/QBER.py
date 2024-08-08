from wolframclient.evaluation import WolframLanguageSession
import pickle

s = """
ClientLibrary`SetErrorLogLevel[];
Latt = 20; 
t0 = 1; 
eta = (1 - pc)/E^(L0/Latt); 
ee = 1 - eta; 
ex = 0.5*eta*e; 
ey = 0.5*eta*e; 
ez = 0.5*eta*e; 
ei = eta*(1 - 1.5*e); 
q00 = Sum[
   If[2*w == m - a, Multinomial[a, w, m - a - w]*ee^a*(ex + ey)^w*
           (ez + ei)^(m - a - w), 0], {a, 1, m}, {w, 0, m - a}]; 
q0p = Sum[
   If[2*w < m - a, Multinomial[a, w, m - a - w]*ee^a*(ex + ey)^w*
           (ez + ei)^(m - a - w), 0], {a, 1, m}, {w, 0, m - a}]; 
q0m = Sum[
   If[2*w > m - a, Multinomial[a, w, m - a - w]*ee^a*(ex + ey)^w*
           (ez + ei)^(m - a - w), 0], {a, 1, m}, {w, 0, m - a}]; 
qp0 = Sum[
   If[Mod[v, 2] == 0 && 2*(b + c) == m, 
    Multinomial[v - c, c, b, m - v - b]*ex^b*
           ey^c*ez^(v - c)*ei^(m - b - v), 0], {v, 0, m}, {c, 0, 
    v}, {b, 0, m - v - c}]; 
qpp = Sum[
   If[Mod[v, 2] == 0 && 2*(b + c) < m, 
    Multinomial[v - c, c, b, m - v - b]*ex^b*ey^c*
           ez^(v - c)*ei^(m - b - v), 0], {v, 0, m}, {c, 0, v}, {b, 
    0, m - v - c}]; 
qpm = Sum[
   If[Mod[v, 2] == 0 && 2*(b + c) > m, 
    Multinomial[v - c, c, b, m - v - b]*ex^b*ey^c*
           ez^(v - c)*ei^(m - b - v), 0], {v, 0, m}, {c, 0, v}, {b, 
    0, m - v - c}]; 
qm0 = Sum[
   If[Mod[v, 2] == 1 && 2*(b + c) == m, 
    Multinomial[v - c, c, b, m - v - b]*ex^b*
           ey^c*ez^(v - c)*ei^(m - b - v), 0], {v, 1, m}, {c, 0, 
    v}, {b, 0, m - v - c}]; 
qmp = Sum[
   If[Mod[v, 2] == 1 && 2*(b + c) < m, 
    Multinomial[v - c, c, b, m - v - b]*ex^b*ey^c*
           ez^(v - c)*ei^(m - b - v), 0], {v, 1, m}, {c, 0, v}, {b, 
    0, m - v - c}]; 
qmm = Sum[
   If[Mod[v, 2] == 1 && 2*(b + c) > m, 
    Multinomial[v - c, c, b, m - v - b]*ex^b*ey^c*
           ez^(v - c)*ei^(m - b - v), 0], {v, 1, m}, {c, 0, v}, {b, 
    0, m - v - c}]; 
Unprotect[Power]; 
(0 | 0.)^(0 | 0.) = 1; 
Protect[Power]; 
p00 = Sum[If[a + b + c >= 1 && 2*(b + w) == n - a - v, 
         
    Multinomial[a, b, c, v, w, n - (a + b + c + v + w)]*q00^a*qp0^b*
     qm0^c*(q0p + q0m)^v*
           (qpp + qmp)^w*(qpm + qmm)^(n - (a + b + c + v + w)), 
    0], {a, 0, n}, {b, 0, n - a}, 
       {c, 0, n - a - b}, {v, 0, n - a - b - c}, {w, 0, 
    n - a - b - c - v}]; 
pp0 = Sum[If[a + b + c >= 1 && 2*(b + w) > n - a - v, 
         
    Multinomial[a, b, c, v, w, n - (a + b + c + v + w)]*q00^a*qp0^b*
     qm0^c*(q0p + q0m)^v*
           (qpp + qmp)^w*(qpm + qmm)^(n - (a + b + c + v + w)), 
    0], {a, 0, n}, {b, 0, n - a}, 
       {c, 0, n - a - b}, {v, 0, n - a - b - c}, {w, 0, 
    n - a - b - c - v}]; 
pm0 = Sum[If[a + b + c >= 1 && 2*(b + w) < n - a - v, 
         
    Multinomial[a, b, c, v, w, n - (a + b + c + v + w)]*q00^a*qp0^b*
     qm0^c*(q0p + q0m)^v*
           (qpp + qmp)^w*(qpm + qmm)^(n - (a + b + c + v + w)), 
    0], {a, 0, n}, {b, 0, n - a}, 
       {c, 0, n - a - b}, {v, 0, n - a - b - c}, {w, 0, 
    n - a - b - c - v}]; 
p0p = Sum[If[Mod[g + h + i, 2] == 0 && 2*(e + h) == n - d - g, 
         
    Multinomial[d, e, g, h, i, n - (d + e + g + h + i)]*q0p^d*qpp^e*
           qmp^(n - (d + e + g + h + i))*q0m^g*qpm^h*qmm^i, 0], {d, 
    0, n}, {e, 0, n - d}, 
       {g, 0, n - d - e}, {h, 0, n - d - e - g}, {i, 0, 
    n - d - e - g - h}]; 
ppp = Sum[If[Mod[g + h + i, 2] == 0 && 2*(e + h) > n - d - g, 
         
    Multinomial[d, e, g, h, i, n - (d + e + g + h + i)]*q0p^d*qpp^e*
           qmp^(n - (d + e + g + h + i))*q0m^g*qpm^h*qmm^i, 0], {d, 
    0, n}, {e, 0, n - d}, 
       {g, 0, n - d - e}, {h, 0, n - d - e - g}, {i, 0, 
    n - d - e - g - h}]; 
pmp = Sum[If[Mod[g + h + i, 2] == 0 && 2*(e + h) < n - d - g, 
         
    Multinomial[d, e, g, h, i, n - (d + e + g + h + i)]*q0p^d*qpp^e*
           qmp^(n - (d + e + g + h + i))*q0m^g*qpm^h*qmm^i, 0], {d, 
    0, n}, {e, 0, n - d}, 
       {g, 0, n - d - e}, {h, 0, n - d - e - g}, {i, 0, 
    n - d - e - g - h}]; 
p0m = Sum[If[Mod[g + h + i, 2] == 1 && 2*(e + h) == n - d - g, 
         
    Multinomial[d, e, g, h, i, n - (d + e + g + h + i)]*q0p^d*qpp^e*
           qmp^(n - (d + e + g + h + i))*q0m^g*qpm^h*qmm^i, 0], {d, 
    0, n}, {e, 0, n - d}, 
       {g, 0, n - d - e}, {h, 0, n - d - e - g}, {i, 0, 
    n - d - e - g - h}]; 
ppm = Sum[If[Mod[g + h + i, 2] == 1 && 2*(e + h) > n - d - g, 
         
    Multinomial[d, e, g, h, i, n - (d + e + g + h + i)]*q0p^d*qpp^e*
           qmp^(n - (d + e + g + h + i))*q0m^g*qpm^h*qmm^i, 0], {d, 
    0, n}, {e, 0, n - d}, 
       {g, 0, n - d - e}, {h, 0, n - d - e - g}, {i, 0, 
    n - d - e - g - h}]; 
pmm = Sum[If[Mod[g + h + i, 2] == 1 && 2*(e + h) < n - d - g, 
         
    Multinomial[d, e, g, h, i, n - (d + e + g + h + i)]*q0p^d*qpp^e*
           qmp^(n - (d + e + g + h + i))*q0m^g*qpm^h*qmm^i, 0], {d, 
    0, n}, {e, 0, n - d}, 
       {g, 0, n - d - e}, {h, 0, n - d - e - g}, {i, 0, 
    n - d - e - g - h}]; 
Psucc = (ppp + ppm + pmp + pmm)^Ns; 
Q = 0.25*(1 - ((ppp + ppm - pmp - pmm)/(ppp + ppm + pmp + pmm))^
       Ns) + 
       0.25*(1 - ((ppp - ppm + pmp - pmm)/(ppp + ppm + pmp + pmm))^
       Ns); 
h[Q_] := (-Q)*Log2[Q] - (1 - Q)*Log2[1 - Q]; 
       """


def QBER(m, n, pc, e, L0, Nhop, session):
    ss = "m=" + str(m) + ";" + "n=" + str(n) + ";" + "pc=" + str(pc) + ";" + "e=" + str(e) + ";" + "Ns=" + str(
        Nhop) + ";" + "L0=" + str(L0) + ";" + s
    session.evaluate(ss)
    return session.evaluate_many(["Max[Psucc*(1 - 2*h[Q]), 0]", "p00+p0p+p0m+pp0+pm0"])


if __name__ == "__main__":
    Nhop = 1
    L0 = 1
    p_loss = [0.0, 0.05]
    p_depo = [0.004, 0.008, 0.012, 0.016, 0.02]
    m_range = [3, 15]
    n_range = [3, 23]
    mn_step = 4
    session = WolframLanguageSession()
    session.start()
    coeff1 = 1
    coeff2 = 1
    p_loss[1] *= coeff1
    for i in range(len(p_depo)):
        p_depo[i] *= coeff2

    para_space = [(m, n, pc, e)
                  for m in range(m_range[0], m_range[1], mn_step)
                  for n in range(n_range[0], n_range[1], mn_step)
                  for pc in p_loss
                  for e in p_depo]
    qber_dict = dict()
    print("Generating")
    i = 0
    for m, n, pc, e in para_space:
        i += 1
        qber_dict[(m, n, pc, e)] = QBER(m, n, pc, e, L0, Nhop, session)
        if i % 50 == 0:
            print(i, "of", len(para_space))
    fname = ("STEP" + str(mn_step) + "LOSS" + str(p_loss) + "DEPO" + str(
        p_depo) +
             "L0" + str(L0) + "NHOP" + str(Nhop) + ".dict")

    session.terminate()
    with open(fname, "wb") as f:
        pickle.dump(qber_dict, f)
