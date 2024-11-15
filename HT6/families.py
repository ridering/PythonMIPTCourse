import enum


class Gender(enum.Enum):
    MALE = 1
    FEMALE = 0

    def __str__(self) -> str:
        return self.name


class Person:
    def __init__(self, gender: Gender, first_name: str,
                 last_name: str, middle_name: str = None) -> None:
        assert isinstance(gender, Gender)
        assert isinstance(first_name, str)
        assert isinstance(last_name, str)
        assert isinstance(middle_name or '', str)

        self._first_name: str = first_name
        self._middle_name: str = middle_name
        self._last_name: str = last_name
        self._gender: Gender = gender

        self._partner: Person = None
        self._father: Person = None
        self._mother: Person = None
        self._children: set[Person] = set()

    def __str__(self) -> str:
        return f"{self.first_name} {self.middle_name or ' - '} {self.last_name}"

    @staticmethod
    def child_from_parents(gender: Gender, first_name: str,
                           mother: 'Person', father: 'Person') -> 'Person':
        assert isinstance(father, Person)
        assert isinstance(mother, Person)

        assert father is mother.partner
        assert mother is father.partner

        child = Person(gender, first_name, father.last_name,
                       father.first_name)
        child._father = father
        child._mother = mother
        father._children.add(child)
        mother._children.add(child)

        return child

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.middle_name or ' - '} {self.last_name}"

    @property
    def first_name(self) -> str:
        return self._first_name

    @property
    def middle_name(self) -> str:
        return self._middle_name

    @property
    def last_name(self) -> str:
        return self._last_name

    @property
    def gender(self) -> Gender:
        return self._gender

    @property
    def partner(self) -> str:
        return self._partner

    @property
    def father(self) -> str:
        return self._father

    @property
    def mother(self) -> str:
        return self._mother

    @property
    def children(self) -> set['Person']:
        return self._children.copy()

    @property
    def personal_data(self) -> dict:
        return {
            'first_name': str(self.first_name),
            'middle_name': str(self.middle_name or '-'),
            'last_name': str(self.last_name),
            'gender': str(self.gender),
            'partner': str(self.partner or '-'),
            'father': str(self.father or '-'),
            'mother': str(self.mother or '-'),
            'children': [str(child) for child in self.children],
        }


class Family:
    def __init__(self) -> None:
        self._members: set[Person] = set()

    def __str__(self) -> str:
        return str([str(member) for member in self.members])

    @staticmethod
    def family_from_marriage(man: Person, woman: Person) -> 'Family':
        assert isinstance(man, Person)
        assert isinstance(woman, Person)

        assert man.gender == Gender.MALE
        assert woman.gender == Gender.FEMALE

        assert man.partner is None
        assert woman.partner is None

        family = Family()
        family._members.add(man)
        family._members.add(woman)
        woman._last_name = man.last_name
        man._partner = woman
        woman._partner = man

        return family

    @property
    def members(self) -> set[Person]:
        return self._members.copy()

    @property
    def all_data(self) -> dict:
        return {'members': [member.personal_data for member in self.members]}


class City:
    def __init__(self, name: str) -> None:
        assert isinstance(name, str)

        self._name: str = name
        self._families: set[Family] = set()

    def __str__(self) -> str:
        return str([str(family) for family in self.families])

    def add_family(self, family: Family) -> None:
        assert isinstance(family, Family)

        self._families.add(family)

    @property
    def name(self) -> str:
        return self._name

    @property
    def families(self) -> set[Family]:
        return self._families.copy()

    @property
    def all_data(self) -> dict:
        return {
            'name': self.name,
            'families': [family.all_data for family in self.families],
        }
