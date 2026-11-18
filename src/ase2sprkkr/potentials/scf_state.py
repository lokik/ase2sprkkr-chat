"""SCF convergence state shared by ordinary and embedded potentials."""


class PotentialScfState:
    def is_scf_converged(self, atoms=None):
        """Return whether this potential represents a converged SCF state."""
        status = self.SCF_INFO.SCFSTATUS()
        if status == "CONVERGED":
            return True
        if status != "START":
            return False
        if atoms is None:
            atoms = self._atoms
        return atoms is not None and atoms.sprkkr_has_converged_data()
