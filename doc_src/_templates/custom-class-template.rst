{{ name | escape | underline}}


.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:

   .. rubric:: {{ _('Class hierarchy') }}
   {% if not objname.startswith('_') %}
   .. inheritance-diagram:: {{ fullname }}
      :parts: 2
   {% endif %}

   |

   .. rubric:: {{ _('Constructor') }}
   {% block init %}
   .. automethod:: __init__
   {% endblock %}

   {% block own_attributes %}
   {% if own_attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :template: custom-base-template.rst
     {% for item in own_attributes %}
        ~{{ name }}.{{ item }}
     {%- endfor %}

   {% endif %}
   {% endblock %}

   {% block own_methods %}
   {% if own_methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :template: custom-base-template.rst
     {% for item in own_methods %}
        ~{{ name }}.{{ item }}
     {%- endfor %}

   {% endif %}
   {% endblock %}

