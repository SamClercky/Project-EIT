#ifndef PI_H
#define	PI_H

#include <xc.h> // include processor files - each processor file is guarded.

#ifdef	__cplusplus
extern "C" {
#endif /* __cplusplus */

    void PI(void);
    
    int PI_GetSensorHeight(void);
    
    void PI_SetSetpoint(int value);
    
    void PI_SetKp(float value);
    
    void PI_SetKi(float value);

#ifdef	__cplusplus
}
#endif /* __cplusplus */

#endif	/* PI_H */

