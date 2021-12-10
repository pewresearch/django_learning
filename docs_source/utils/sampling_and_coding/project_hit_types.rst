Project HIT types
------------------

Project HIT types specify a type of task to be completed. They primarily contain Mechanical Turk parameters
specifying how much a task should cost, and other things like how long HITs should remain active until they expire.
They're specified in JSON files.

.. code:: python

    {
      "title": "Label some documents",
      "description": "Label some documents and get paid!",
      "keywords": ["labeling"],
      "price": 0.1,
      "approval_wait_hours": 24,
      "duration_minutes": 10,
      "lifetime_days": 7,
      "min_approve_pct": 0.95,
      "min_approve_cnt": 100
    }

The contain the following parameters:

    * ``title``: short name to be given to the task to be performed
    * ``description``: more verbose description
    * ``keywords``: search terms that Turkers can use to find the task
    * ``price``: price per HIT to be paid, in dollars
    * ``approval_wait_hours``: how many hours to wait before auto-approving completed tasks
    * ``duration_minutes``: maximum number of minutes Turkers have to complete a single task
    * ``lifetime_days``: maximum number of days uncompleted HITs will remain available after creation
    * ``min_approve_pct``: minimum approval percentage for workers to qualify for the HITs
    * ``min_approve_cnt``: minimum number of good HITs workers must have done to qualify

HIT types get synced with the database and can be found in the ``django_learning.models.HITType`` model:

.. code:: python

    from django_learning.models import HITType

    >>> HITType.objects.all()

    <BasicExtendedManager [<HITType: my_project: my_hit_type>]>

Running ``.save()`` on a ``HITType`` instance will re-sync it with the configuration file.

You can also access HIT type configurations directly through Django Learning utils:

.. code:: python

    from django_learning.utils.project_hit_types import project_hit_types

    >>> project_hit_types["my_hit_type"]()

    {
      "title": "Label some documents",
      "description": "Label some documents and get paid!",
      "keywords": ["labeling"],
      "price": 0.1,
      "approval_wait_hours": 24,
      "duration_minutes": 10,
      "lifetime_days": 7,
      "min_approve_pct": 0.95,
      "min_approve_cnt": 100
    }