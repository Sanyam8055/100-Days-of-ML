import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';

void main() {
  runApp(
    Myapp());
}
 class Myapp extends StatelessWidget {
   @override
   Widget build(BuildContext context) {
     return MaterialApp(
       home: Scaffold(
         backgroundColor: Colors.deepPurpleAccent,
         body: SafeArea(
           child:Column(
             mainAxisAlignment: MainAxisAlignment.center,
             children: <Widget>[
               CircleAvatar(
                 radius:60,

                 backgroundImage: AssetImage('images/sanyam.jpeg'),
               ),
               Text(
                 'Sanyam Singh',
                 style:TextStyle(
                   fontFamily: 'Pacifico',
                   color: Colors.white,
                   fontSize: 30,
                   fontWeight: FontWeight.bold,
                 ),
               ),
               Text(
                   'FLUTTER DEVELPOER',
                   style:TextStyle(
                     fontFamily: 'SourceSansPro',
                     color: Colors.blue.shade100,
                     fontSize: 20,
                     fontWeight: FontWeight.bold,
                   ),
               ),
               SizedBox(
                 height: 20,
                 width:150.0,
                 child: Divider(
                   color:Colors.white,
                 ),
               ),
               Card(
                 color: Colors.white,
                 margin: EdgeInsets.symmetric(vertical: 10,horizontal: 25),
                 child: ListTile(
                   leading:Icon(
                       Icons.phone,
                       size:20,
                       color: Colors.teal.shade900,
                       ),
                       title:Text(
                         '+917607016555',
                         style: TextStyle(
                         fontFamily: 'Source Sans Pro',
                         fontSize: 20,
                           color: Colors.teal.shade900,
                       ),
                       ),
                   ),
                 ),

               Card(
                 color: Colors.white,
                 margin: EdgeInsets.symmetric(vertical: 10,horizontal: 25),
                 child: ListTile(
                   leading: Icon(
                     Icons.email,
                       color: Colors.teal.shade900,
                     ),
                     title:Text(
                       'sanyam12sks@gmail.com',
                       style: TextStyle(
                         color: Colors.teal.shade900,
                         fontFamily: 'Source Sans Pro',
                         fontSize: 20,
                       ),
                     ),
                 ),
               ),
               Card(
                 color: Colors.white,
                 margin: EdgeInsets.symmetric(vertical: 10,horizontal: 25),
                 child: ListTile(
                   leading: Icon(
                     Icons.web,
                     color: Colors.teal.shade900,
                   ),
                   title:Text(
                     'https://sanyamsingh.ml/',
                     style: TextStyle(
                       color: Colors.teal.shade900,
                       fontFamily: 'Source Sans Pro',
                       fontSize: 20,
                     ),
                   ),
                 ),
               ),
             ],
           ),
         ),
       ),
     );
   }
 }
