from assert_testing.content.assert_exception import divide



def test_divide():

    assert divide(5) == 2
    assert isinstance(divide(5),float) # no exception took place
    assert divide(0) == ZeroDivisionError.__name__
    assert divide('number') == TypeError.__name__

"""
solution to relative imports https://stackoverflow.com/questions/16981921/relative-imports-in-python-3
solution to assert errors https://docs.pytest.org/en/6.2.x/assert.html
solution to non raised exceptions: https://stackoverflow.com/questions/6181555/pass-a-python-unittest-if-an-exception-isnt-raised/29125944
-> #Simply call your functionality, e.g. do_something(). If an unhandled exception gets raised, the test automatically fails! There is really no reason to do anything else. This is also the reason why assertDoesNotRaise() does not exist.


def test_zero_division():
    with pytest.raises(ZeroDivisionError):
        1 / 0
        
def test_recursion_depth():
    with pytest.raises(RuntimeError) as excinfo:

        def f():
            f()

        f()
    assert "maximum recursion" in str(excinfo.value)


pytest.raises(ExpectedException, func, *args, **kwargs)

"""
