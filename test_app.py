from app import add, divide

def test_add():
    assert add(2, 3) == 5

def test_divide():
    assert divide(10, 2) == 5

def test_fail_case():
    assert add(2, 2) == 5   #for demo