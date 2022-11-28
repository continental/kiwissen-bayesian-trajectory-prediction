{{ fullname | escape | underline }}
    
.. rubric:: Description

.. automodule:: {{ fullname }}
   
    {% block modules %}
    {% if modules %}
    .. rubric:: Sub-modules

    .. autosummary::
        :toctree:
        :recursive:
        {% for submodule in modules %}
        ~{{ submodule }}
        {% endfor %}

    {% endif %}
    {% endblock %}


    {% block attributes %}
    {% if attributes %}
    .. rubric:: Attributes

    .. autosummary::
        :toctree:
        {% for item in attributes %}
        ~{{ item }}
        {%- endfor %}

    {% endif %}
    {% endblock %}


    {% block classes %}
    {% if classes %}
    .. rubric:: Classes

    .. autosummary::
        :toctree:
        :nosignatures:
        {% for class in classes %}
        ~{{ class }}
        {% endfor %}

    {% endif %}
    {% endblock %}


    {% block exceptions %}
    {% if exceptions %}
    .. rubric:: Exceptions

    .. autosummary::
        :nosignatures:
        :toctree:
        {% for class in exceptions %}
        ~{{ class }}
        {% endfor %}

    {% endif %}
    {% endblock %}


    {% block functions %}
    {% if functions %}
    .. rubric:: Functions

    .. autosummary::
        :toctree:
        {% for function in functions %}
        ~{{ function }}
        {% endfor %}

    {% endif %}
    {% endblock %}