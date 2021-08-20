from assert_testing.content.assert_exception import divide



def test_divide():

    assert divide(5) == 2
    assert isinstance(divide(5),float)
    assert divide(0) == ZeroDivisionError
    assert isinstance(divide(0),ZeroDivisionError)


    ### solution to relative imports https://stackoverflow.com/questions/16981921/relative-imports-in-python-3
    ### solution to assert errors https://docs.pytest.org/en/6.2.x/assert.html