"""
A Sphinx example and cheat sheet.
 
Comments
########
 
.. comment This is a comment!
 
.. code-block:: python
 
    .. comment This is a comment!
 
Basic Syntax
############
 
*italic*
 
**bold**
 
``verbatim with special characters such as *, also useful for inline code examples``
 
.. code-block:: python
 
    *italic*
 
    **bold**
    
    ``verbatim with special characters such as *, also useful for inline code examples``
 
Links and Downloads
###################
 
Hyperlink: `David Stutz <https://davidstutz.de>`_
 
Download: :download:`__init__.py <../../module/__init__.py>`
 
.. code-block:: python
 
    Hyperlink: `David Stutz <https://davidstutz.de>`_
    
    Download: :download:`__init__.py <../../module/__init__.py>`
 
Headings
########
 
Level 2
*******
 
Level 3
=======
 
Level 4
-------
 
Level 5
^^^^^^^
 
Note that level 5 does not impose any styling, but is added to the table of contents
on the left.
 
.. code-block:: python
 
    Headings
    ########
    
    Level 2
    *******
    
    Level 3
    =======
    
    Level 4
    -------
    
    Level 5
    ^^^^^^^
 
Lists
#####
 
* Item 1
* Item 2
* Item 3
 
* Multi-
  line item
 
1. Item 1
2. Item 2
3. Item 3
 
#. Item 4
#. Item 5
 
.. code-block:: python
 
    * Item 1
    * Item 2
    * Item 3
    
    * Multi-
      line item
    
    1. Item 1
    2. Item 2
    3. Item 3
    
    #. Item 4
    #. Item 5
 
Tables
######
 
Complex variant:
 
+------------+------------+
| Header 1   | Header 2   |
+============+============+
| Cell 1.1   | Cell 1.2   |
+------------+------------+
| Multi-column            |
+------------+------------+
| Cell 3.1   | Multi-row  |
+------------+            |
| Cell 4.1   |            |
+------------+------------+
 
Another, simpler variant:
 
======== ========
Header 1 Header 2
======== ========
Cell 1.1 Cell 1.2
Cell 2.1 Cell 2.2
======== ========
 
.. code-block:: python
 
    +------------+------------+
    | Header 1   | Header 2   |
    +============+============+
    | Cell 1.1   | Cell 1.2   |
    +------------+------------+
    | Multi-column            |
    +------------+------------+
    | Cell 3.1   | Multi-row  |
    +------------+            |
    | Cell 4.1   |            |
    +------------+------------+
    
    ======== ========
    Header 1 Header 2
    ======== ========
    Cell 1.1 Cell 1.2
    Cell 2.1 Cell 2.2
    ======== ========
 
Boxes
#####
 
.. seealso:: See also box ...
.. todo:: To do box ...
.. warning:: Warning box ...
 
Code
####
 
Simple code box::
 
    print('done ...')
 
Line numbers and language option:
 
.. code-block:: python
    :linenos:
    
    print('done ...')
 
Math
####
 
Note that backslashes need to be escaped!
 
If the math isn't rendered directly, refresh using Shift + F5 or Ctrl + Shift + R (in most browsers).
 
Inline: :math:`\\alpha`
 
Block:
 
.. math::
 
    \\sum_{i = 1}^n w_i x_i
 
.. code-block:: latex
    
    Inline: :math:`\\alpha`
    
    Block:
    
    .. math::
    
        \\sum_{i = 1}^n w_i x_i
 
Images and Figures
##################
 
Image:
 
.. image:: images/pocoo.jpg
 
Figure:
 
.. figure:: images/pocoo.jpg
    :align: center
    :alt: Pocoo
    :figclass: align-center
    
    Pocoo figure ...
 
.. code-block:: python
 
    Image:
 
    .. image:: images/pocoo.jpg
    
    Figure:
    
    .. figure:: images/pocoo.jpg
        :align: center
        :alt: Pocoo
        :figclass: align-center
        
        Pocoo figure ...
 
Misc Elements
#############
 
Topic:
 
.. topic:: My Topic
 
    My topic text ...
 
Sidebar:
 
.. sidebar:: My Sidebar
    
    My sidebar text ...
    
.. code-block:: python
 
    .. topic:: My Topic
    
        My topic text ...
    
    .. sidebar:: My Sidebar
        
        My sidebar text ...
        
Citations
#########
 
Citation in text [Stutz2015]_
 
.. [Stutz2015] D. Stutz. Superpixel Segmentation: An Evaluation. GCPR, 2015.
 
.. code-block:: python
 
    Citation in text [Stutz2015]_
    
    .. [Stutz2015] D. Stutz. Superpixel Segmentation: An Evaluation. GCPR, 2015.
 
Footnotes
#########
 
The footnote section needs to be added at the end ...
 
.. code-block:: python
    
    Footnote [#f]_
    
    .. comment:: ...
    
    .. rubric:: Footnotes
    
    .. [#f] Footenote text ...
 
Footnote [#f]_
 
.. rubric:: Footnotes
 
.. [#f] Footenote text ...
"""
 
class AClass:
    """
    Class docstring, with reference to the :mod:`module`, or another class
    :class:`module.AnotherClass` and its function :func:`module.AnotherClass.foo`.
    """
 
class AnotherClass:
    """
    Another class' docstring.
    """
    
    def foo(arg1, arg2):
        """
        A method's docstring with parameters and return value.
        
        Use all the cool Sphinx capabilities in this description, e.g. to give
        usage examples ...
        
        :Example:
 
        >>> another_class.foo('', AClass())        
        
        :param arg1: first argument
        :type arg1: string
        :param arg2: second argument
        :type arg2: :class:`module.AClass`
        :return: something
        :rtype: string
        :raises: TypeError
        """
        
        return '' + 1
        
def foo(arg1, arg2):
    """
    A method's docstring with parameters and return value.

    Use all the cool Sphinx capabilities in this description, e.g. to give
    usage examples ...

    :Example:

    >>> another_class.foo('', AClass())        
        
    :param arg1: first argument
    :type arg1: string
    :param arg2: second argument
    :type arg2: :class:`module.AClass`
    :return: something
    :rtype: string
    :raises: TypeError
    """
    
    return '' + 1