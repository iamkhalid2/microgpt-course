// Small datasets for the mutation lab. One entry per line, letters only.
const words = (s) => s.split(/\s+/).map((w) => w.toLowerCase().replace(/[^a-z]/g, '')).filter(Boolean);

export const DINOSAURS = words(`tyrannosaurus velociraptor triceratops stegosaurus brachiosaurus diplodocus allosaurus ankylosaurus spinosaurus iguanodon
parasaurolophus pachycephalosaurus carnotaurus giganotosaurus apatosaurus brontosaurus compsognathus deinonychus dilophosaurus gallimimus ceratosaurus
coelophysis megalosaurus utahraptor therizinosaurus oviraptor protoceratops maiasaura hadrosaurus edmontosaurus corythosaurus lambeosaurus styracosaurus
centrosaurus pentaceratops torosaurus albertosaurus gorgosaurus daspletosaurus tarbosaurus baryonyx suchomimus irritator carcharodontosaurus acrocanthosaurus
microraptor archaeopteryx troodon sinosauropteryx mamenchisaurus argentinosaurus patagotitan dreadnoughtus camarasaurus nigersaurus amargasaurus plateosaurus
massospondylus herrerasaurus eoraptor staurikosaurus scelidosaurus kentrosaurus huayangosaurus gastonia polacanthus nodosaurus edmontonia euoplocephalus
saichania minmi hypsilophodon dryosaurus camptosaurus tenontosaurus muttaburrasaurus ouranosaurus psittacosaurus yutyrannus guanlong dilong sinraptor
monolophosaurus cryolophosaurus majungasaurus abelisaurus rajasaurus masiakasaurus noasaurus ornitholestes struthiomimus ornithomimus deinocheirus
segnosaurus alxasaurus beipiaosaurus caudipteryx citipati mononykus shuvuuia alvarezsaurus dromaeosaurus saurornitholestes bambiraptor achillobator
austroraptor unenlagia rahonavis anchiornis epidexipteryx scansoriopteryx kulindadromeus leptoceratops montanoceratops zuniceratops chasmosaurus
anchiceratops arrhinoceratops einiosaurus achelousaurus pachyrhinosaurus brachylophosaurus gryposaurus prosaurolophus saurolophus tsintaosaurus
shantungosaurus kritosaurus hypacrosaurus olorotitan charonosaurus amurosaurus nipponosaurus bactrosaurus telmatosaurus rhabdodon zalmoxes`);

export const CITIES = words(`tokyo delhi shanghai saopaulo mexicocity cairo mumbai beijing dhaka osaka newyork karachi buenosaires chongqing istanbul kolkata manila lagos
riodejaneiro tianjin kinshasa guangzhou losangeles moscow shenzhen lahore bangalore paris bogota jakarta chennai lima bangkok seoul nagoya hyderabad london
tehran chicago chengdu nanjing wuhan luanda ahmedabad kualalumpur hongkong hangzhou riyadh baghdad santiago surat madrid pune harbin houston dallas toronto
daressalaam miami belohorizonte singapore philadelphia atlanta fukuoka khartoum barcelona johannesburg saintpetersburg qingdao dalian washington yangon alexandria
jinan guadalajara casablanca nairobi monterrey sydney melbourne berlin rome kyiv vienna warsaw budapest prague lisbon athens dublin oslo stockholm helsinki
copenhagen amsterdam brussels zurich geneva milan naples turin florence venice munich hamburg cologne frankfurt stuttgart lyon marseille toulouse nice bordeaux
seville valencia bilbao porto edinburgh glasgow manchester liverpool birmingham leeds bristol cardiff belfast montreal vancouver calgary ottawa boston seattle
denver phoenix portland detroit cleveland pittsburgh nashville memphis orlando tampa honolulu anchorage havana panama quito lapaz montevideo asuncion caracas
brasilia recife salvador fortaleza curitiba medellin cali cusco accra dakar abidjan tunis algiers tripoli addisababa kampala kigali lusaka harare maputo windhoek
gaborone antananarivo colombo kathmandu thimphu islamabad kabul tashkent almaty bishkek baku tbilisi yerevan amman beirut damascus jerusalem doha muscat kuwait
hanoi hue danang phnompenh vientiane taipei busan incheon sapporo kyoto hiroshima nagasaki kobe auckland wellington christchurch perth brisbane adelaide hobart darwin`);

export const DATASETS = {
  names: { label: 'The original 32,033 names', docs: null },
  dino: { label: 'Dinosaurs', docs: DINOSAURS },
  city: { label: 'World cities', docs: CITIES },
  own: { label: 'My own list…', docs: [] },
};
export const cleanList = (text) => words(text.replace(/[\r,;]+/g, '\n').split('\n').map((l) => l.trim().replace(/\s+/g, '')).join('\n'));
