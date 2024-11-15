import json

from families import *

liza = Person(Gender.FEMALE, 'Liza', 'Borisov', 'Egor')
assert liza.first_name == 'Liza'
assert liza.last_name == 'Borisov'
assert liza.middle_name == 'Egor'
assert liza.gender == Gender.FEMALE

oleg = Person(Gender.MALE, 'Oleg', 'Prokopenko', 'Anton')
assert oleg.gender == Gender.MALE

family1 = Family.family_from_marriage(oleg, liza)
assert liza.last_name == oleg.last_name
assert oleg is liza.partner
assert liza is oleg.partner

vanya = Person.child_from_parents(Gender.MALE, 'Vanya', liza, oleg)
assert vanya.gender == Gender.MALE
assert vanya.first_name == 'Vanya'
assert vanya.last_name == oleg.last_name
assert vanya.middle_name == oleg.first_name
assert oleg is vanya.father
assert liza is vanya.mother
assert vanya in oleg.children
assert vanya in liza.children

sveta = Person.child_from_parents(Gender.FEMALE, 'Sveta', liza, oleg)
assert sveta in liza.children
assert sveta in oleg.children
assert vanya in liza.children
assert vanya in oleg.children


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
