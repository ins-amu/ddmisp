import numpy as np

class ClassEval:
    def __init__(self, true, est):
        true = np.array(true, dtype=bool)
        est  = np.array(est,  dtype=bool)

        self.p = np.sum(true)                               #: Positive
        self.n = np.sum(~true)                              #: Negative
        self.pop = len(true)                                #: Total population

        self.tp = np.sum(true & est)                        #: True positive
        self.fp = np.sum((~true) & est)                     #: False positive
        self.tn = np.sum((~true) & (~est))                  #: True negative
        self.fn = np.sum(true & (~est))                     #: False negative

        self.tpr = self.sens = self.tp/self.p               #: True positive rate, or Sensitivity
        self.tnr = self.spec = self.tn/self.n               #: True negative rate, or Specificity
        self.ppv = self.prec = self.tp/(self.tp + self.fp)  #: Positive predictive value, or Precision
        self.npv = self.tn/(self.tn + self.fn)              #: Negative predictive value

        self.acc = (self.tp + self.tn)/(self.p + self.n)      #: Accuracy
        self.f1 = 2*self.tp / (2*self.tp + self.fp + self.fn) #: F1 score

    def _repr_pretty_(self, p, cycle=False):
        p.text("ClassEval (p=%d, n=%d, p+n=%d)\n" % (self.p, self.n, self.pop))
        p.text("TPR (Sensitivity)  %.2f\n" % (self.tpr))
        p.text("TNR (Specificity)  %.2f\n" % (self.tnr))
        p.text("PPV (Precision)    %.2f\n" % (self.ppv))
        p.text("Accuracy           %.2f\n" % (self.acc))
        p.text("F1                 %.2f\n" % (self.f1))
