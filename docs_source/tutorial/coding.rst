# Coding

## Coding in-house

## Coding on Mechanical Turk


In-House Coding
===============

1. Now that you have a sample extracted, you need to actually create coding assignments (HITs). For in-house coding, run::

    # do not include --template_name if using the default template
    python manage.py run_command create_sample_hits_experts <project name> <sample name> --num_coders <number coders per question> --template_name <custom HTML template filename, without extension>

2. Finally, you can access the sample coding interface via the url ``/learning/project/<project name>/sample/<sample name>``.

MTurk Coding
============

1. Now that you have a sample extracted, you need to actually create coding assignments (HITs) on mturk::

    # do not include --template_name if using the default template
    python manage.py run_command create_sample_hits_mturk <project name> <sample name> --prod --num_coders <number coders per question> --template_name <custom HTML template filename, without extension>

2. Until all hits are completed, you will need to pull down results periodically. You can additionally add ``--loop`` to sync every 30 seconds (unless a different time is supplied with ``--time_sleep``)::

    python manage.py run_command sync_sample_hits_mturk <project name> <sample name> --prod

3.  After syncing, you must also approve results so that the turkers get paid. It's wise to only approve a fraction of all complete-but-unapproved results at one time, so that turkers get the impression we are paying attention! Of course, you should  actually review some hits too!

    ::

        # probability is a 0-1.0 float, where e.g. .1 means 10% of
        # complete-but-unapproved results will randomly be approved
        python manage.py run_command approve_sample_hits_mturk <project name> <sample name> --prod --probability .1

