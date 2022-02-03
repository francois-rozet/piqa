{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
   :members:

   {% block members %}
   {% if attributes or classes or functions %}
   Members
   -------

   .. autosummary::
      :nosignatures:
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% for item in functions %}
      {{ item }}
   {%- endfor %}

   {% endif %}
   {% endblock %}

{% block modules %}
{% if modules %}
Submodules
----------

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}

{% endif %}
{% endblock %}
