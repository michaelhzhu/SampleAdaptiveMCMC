module Models

using Distributions
using AtomsM
import Model_LinearReg, Model_LogisticReg
export print_errors


print_errors(data::Model_LinearReg.LinearReg, atoms::Atoms) = Model_LinearReg.print_errors(data, atoms)
print_errors(data::Model_LogisticReg.LogisticReg, atoms::Atoms) = Model_LogisticReg.print_errors(data, atoms)


end