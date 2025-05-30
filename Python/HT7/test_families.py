import json

from families import *

try:

    liza = Person(Gender.FEMALE, 'Liza', 'Borisov', 'Egor')
    assert liza.first_name == 'Liza', 'firstname is not Liza'
    assert liza.last_name == 'Borisov', 'lastname is not Borisov'
    assert liza.middle_name == 'Egor', 'middlename is not Egor'
    assert liza.gender == Gender.FEMALE, 'Gender is not Female'

    oleg = Person(Gender.MALE, 'Oleg', 'Prokopenko', 'Anton')
    assert oleg.gender == Gender.MALE, 'gende is not male'

    family1 = Family.family_from_marriage(oleg, liza)
    assert liza.last_name == oleg.last_name, 'lastnames doesn\'t equal'
    assert oleg is liza.partner, 'oleg is not liza\'s partner'
    assert liza is oleg.partner, 'liza is not oleg\'s partner'

    vanya = Person.child_from_parents(Gender.MALE, 'Vanya', liza, oleg)
    assert vanya.gender == Gender.MALE, 'gender is not male'
    assert vanya.first_name == 'Vanya', 'firstname is not vanya'
    assert vanya.last_name == oleg.last_name, 'lastname is not oleg\'s lastname'
    assert vanya.middle_name == oleg.first_name, 'oleg is not liza\'s partner'
    assert oleg is vanya.father, 'oleg is not vanya\'s father'
    assert liza is vanya.mother, 'liza is not vanya\'s mother'
    assert vanya in oleg.children, 'vanya is not in oleg\'s children'
    assert vanya in liza.children, 'vanya is not in liza\'s children'

    sveta = Person.child_from_parents(Gender.FEMALE, 'Sveta', liza, oleg)
    assert sveta in liza.children, 'sveta is not in ilza\'s children'
    assert sveta in oleg.children, 'sveta is not in oleg\'s children'
    assert vanya in liza.children, 'vanya is not in liza\'s children'
    assert vanya not in oleg.children, 'vanya is not in oleg\'s children'


    nina = Person(Gender.FEMALE, 'Nina', 'Kazantsev')
    eugene = Person(Gender.MALE, 'Eugene', 'Kondratiev', 'Semen')
    family2 = Family.family_from_marriage(eugene, nina)
    boris = Person.child_from_parents(Gender.MALE, 'Boris', nina, eugene)
    assert nina.middle_name is None

    family3 = Family.family_from_marriage(boris, sveta)

    manchester = City('Manchester')
    manchester.add_family(family1)
    manchester.add_family(family2)

    liverpool = City('Liverpool')
    liverpool.add_family(family3)

    print(json.dumps(manchester.all_data, indent=2))
    print(json.dumps(liverpool.all_data, indent=2))

except AssertionError as e:
    print(e)