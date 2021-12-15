Project HIT Templates
======================

By default, Django Learning will use the default HIT template that can be found in
``django_learning/templates/django_learning/hit.html``. This template displays the document's text alongside all of the
questions in your project, and contains a bunch of logic for displaying different question types, handling
the visibility of dependencies, adding tooltips and popup modals, and hooking things into Mechanical Turk. If you
want to make modifications to how your coding project gets displayed, you can copy this template into your own
HTML file, located in one of your ``settings.DJANGO_LEARNING_PROJECT_HIT_TEMPLATES`` folders, and make whatever changes
to the layout you'd like. Things we've done in the past include adding images and videos, color-coded annotations to
the document text, embedding tweets, and adding external links to things like news articles and Facebook posts.

