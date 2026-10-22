"""Public live facade for one persisted project."""

from __future__ import annotations


class Project:
    """A project identity with live data and actions backed by its repository.

    Immutable ProjectRecord values stay in the data layer; this facade never
    caches snapshots or duplicates repository persistence policy.
    """

    def __init__(self, repository, project_id):
        self._repository = repository
        self.id = repository.project(project_id).id

    @property
    def record(self):
        return self._repository.project(self.id)

    @property
    def name(self):
        return self.record.name

    @property
    def parent_id(self):
        return self.record.parent_id

    @property
    def generation(self):
        return self.record.generation

    def refresh(self):
        return self.record

    def rename(self, name, *, overwrite=False):
        current = self.record
        return self._repository.save(
            current.rename(name),
            expected_generation=current.generation,
            overwrite=overwrite,
        )

    def move_to(self, parent_id, *, overwrite=False):
        current = self.record
        return self._repository.save(
            current.move_to(parent_id),
            expected_generation=current.generation,
            overwrite=overwrite,
        )

    def jobs(self, *, newest=True):
        return self._repository.jobs(self.id, newest=newest)

    @property
    def last_job(self):
        return self._repository.last_job(self.id)

    def job(self, run_id):
        job = self._repository.job(run_id)
        if job.record.project_id != self.id:
            raise ValueError("Job belongs to another project")
        return job

    def delete(self):
        self._repository.delete(self.id, expected_generation=self.generation)
